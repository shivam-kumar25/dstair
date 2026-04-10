import json
import logging
import os
import requests
from datetime import datetime, timezone

from extensions import db
from models import AIAnalysis, Sphere

logger = logging.getLogger(__name__)


class AIService:
    """
    Handles LLM-backed country evaluation sphere-by-sphere.
    Each sphere is a separate, small LLM call to stay within token limits.
    """

    PROVIDER_CONFIG = {
        'groq': {
            'kind': 'openai_compatible',
            'url': 'https://api.groq.com/openai/v1/chat/completions',
            'model': 'llama-3.3-70b-versatile',
            'max_tokens': 32768,
        },
        'openai': {
            'kind': 'openai_compatible',
            'url': 'https://api.openai.com/v1/chat/completions',
            'model': 'gpt-4o-mini',
            'max_tokens': 16384,
        },
        'claude': {
            'kind': 'anthropic',
            'url': 'https://api.anthropic.com/v1/messages',
            'model': 'claude-haiku-4-5-20251001',
            'anthropic_version': '2023-06-01',
            'max_tokens': 8192,
        },
        'gemini': {
            'kind': 'gemini',
            'url': 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
            'model': 'gemini-2.0-flash-exp',
            'max_tokens': 8192,
        },
        'openrouter': {
            'kind': 'openai_compatible',
            'url': 'https://openrouter.ai/api/v1/chat/completions',
            'model': 'meta-llama/llama-3.3-70b-instruct',
            'max_tokens': 16384,
        },
    }

    # Number of spheres to send in a single LLM request.
    # 3 → 3 requests for 9 spheres.
    SPHERE_BATCH_SIZE = 3

    # Per-request HTTP timeout in seconds.  60 s is enough for most providers
    # to return a 3-sphere response; keeps key-fallback fast when a provider hangs.
    REQUEST_TIMEOUT = 60

    def __init__(self):
        self.env_api_key = os.getenv('GROQ_API_KEY')
        if not self.env_api_key:
            logger.warning("GROQ_API_KEY not set. AI Evaluator will rely on user-saved API keys.")

    # ── Public entry point ───────────────────────────────────────────────────

    def evaluate_country(
        self,
        user_id,
        country_code,
        existing_analysis_id=None,
        additional_instructions=None,
        selected_key_id=None,
        write_lock=None,
        cancel_check=None,
    ):
        all_keys = self._resolve_all_api_keys(user_id, selected_key_id=selected_key_id)
        if not all_keys:
            raise RuntimeError("No API key available. Please configure an API key in the Setup page.")

        orm_spheres = Sphere.get_all_ordered()
        if not orm_spheres:
            raise RuntimeError("Institutional framework (spheres) not found in database.")

        # Snapshot sphere + question data into plain Python objects NOW, before any
        # db.session.commit() calls (which expire all session objects).  This prevents
        # SQLAlchemy lazy-load failures later in the background thread.
        spheres = [
            _SphereSnap(
                name=s.name,
                label=s.label,
                questions=[_QSnap(id=q.id, content=q.content) for q in s.questions],
            )
            for s in orm_spheres
        ]

        # Load or create the analysis record
        if existing_analysis_id:
            analysis = self._fresh_get(existing_analysis_id)
            if not analysis:
                raise RuntimeError(f"AIAnalysis record {existing_analysis_id} disappeared.")
        else:
            analysis = AIAnalysis.get_by_country(country_code)
            if not analysis:
                analysis = AIAnalysis(country=country_code, status='in_progress').save()

        # Persistent log visible in every poll response
        run_log = []

        def push_status(stage, progress, **extra):
            run_log.append(stage)
            self._update_status(analysis.id, 'in_progress', {
                'stage': stage,
                'progress': progress,
                'run_log': run_log[-40:],
                **extra,
            }, write_lock=write_lock)

        def should_stop():
            """Returns True if we should abort — cancelled, timed-out, or DB says error."""
            if cancel_check and cancel_check():
                return True
            # Also check DB state so timer-fired errors abort the background thread
            fresh = self._fresh_get(analysis.id)
            return fresh is not None and fresh.status not in ('in_progress', 'not_started')

        try:
            total_questions = sum(len(list(s.questions)) for s in spheres)
            total_spheres = len(spheres)

            # Group spheres into batches — fewer, larger LLM requests reduce API call
            # overhead and rate-limit exposure.  With 9 spheres: 3 batches of 3.
            batch_size = self.SPHERE_BATCH_SIZE
            batches = [spheres[i:i + batch_size] for i in range(0, len(spheres), batch_size)]
            total_batches = len(batches)

            push_status(
                f'Starting — {total_spheres} spheres, {total_questions} questions '
                f'({total_batches} batch{"es" if total_batches != 1 else ""})',
                3,
                keys_available=len(all_keys),
            )

            all_ratings = {}
            all_comments = {}
            last_raw_sample = {}
            # Cache the last successful key index so we try it first next batch
            preferred_key_idx = 0

            for batch_idx, batch_spheres in enumerate(batches):
                if should_stop():
                    raise RuntimeError("Aborted — evaluation was cancelled or timed out.")

                sphere_names = ', '.join(s.label for s in batch_spheres)
                batch_base_progress = 5 + int((batch_idx / total_batches) * 82)
                push_status(
                    f'Batch {batch_idx + 1}/{total_batches}: {sphere_names}',
                    batch_base_progress,
                )

                payload = self._build_sphere_payload(country_code, batch_spheres, additional_instructions)

                # Re-order keys so the last successful key is tried first
                ordered_keys = (
                    [all_keys[preferred_key_idx]] +
                    [k for i, k in enumerate(all_keys) if i != preferred_key_idx]
                )

                # Capture loop-local values for the closure
                _base = batch_base_progress
                _slice = 82 / total_batches

                def on_provider_status(stage, progress_inner, _b=_base, _s=_slice, **extra):
                    mapped = _b + int((progress_inner - 20) / 45 * _s * 0.8)
                    push_status(
                        stage,
                        max(_b, min(mapped, _b + int(_s))),
                        **extra,
                    )

                batch_ratings, batch_comments, raw_sample, winning_key_idx = \
                    self._evaluate_with_fallback(
                        ordered_keys, payload, batch_spheres,
                        on_status=on_provider_status,
                        cancel_check=should_stop,
                    )

                all_ratings.update(batch_ratings)
                all_comments.update(batch_comments)
                last_raw_sample = raw_sample

                # Map winning_key_idx back to the original key list
                winning_key = ordered_keys[winning_key_idx]
                try:
                    preferred_key_idx = all_keys.index(winning_key)
                except ValueError:
                    preferred_key_idx = 0

            # ── Aggregate ────────────────────────────────────────────────────
            push_status('Computing sphere aggregates', 90)
            sphere_aggregates = {}
            for sphere in spheres:
                sphere_scores = {
                    str(q.id): all_ratings[str(q.id)]
                    for q in sphere.questions
                    if str(q.id) in all_ratings
                }
                sphere_aggregates[sphere.name] = self._calculate_normalized_sphere_avg(sphere_scores)

            # Final write — acquire write_lock so timeout callback can't race
            with _optional_lock(write_lock):
                # Re-fetch to get the latest DB state before writing
                analysis = self._fresh_get(analysis.id)
                if analysis is None:
                    raise RuntimeError("Analysis record disappeared before completion write.")
                if analysis.status not in ('in_progress', 'not_started'):
                    logger.warning(
                        "Skipping mark_completed — status already '%s' for analysis %s",
                        analysis.status, analysis.id,
                    )
                    return analysis.id

                run_log.append('Evaluation complete')
                analysis.mark_completed(
                    scores=all_ratings,
                    comments=all_comments,
                    metadata={
                        'aggregates': sphere_aggregates,
                        'last_run_by_user_id': user_id,
                        'completion_timestamp': datetime.now(timezone.utc).isoformat(),
                        'provider_used': last_raw_sample.get('provider', 'Unknown'),
                        'model_used': last_raw_sample.get('model', ''),
                        'run_log': run_log[-40:],
                    }
                )
            return analysis.id

        except Exception as exc:
            safe_msg = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.error("AI evaluation failure for %s: %s", country_code, safe_msg)
            with _optional_lock(write_lock):
                try:
                    fresh = self._fresh_get(analysis.id)
                    if fresh and fresh.status == 'in_progress':
                        fresh.mark_error(safe_msg)
                except Exception as inner:
                    logger.error("ORM mark_error failed for analysis %s: %s — falling back to raw SQL", analysis.id, inner)
                    # Last resort: raw SQL so the record is never left stuck in in_progress
                    try:
                        db.session.rollback()
                        from sqlalchemy import text
                        db.session.execute(
                            text("UPDATE ai_analyses SET status='error', "
                                 "updated_at=CURRENT_TIMESTAMP WHERE id=:id AND status='in_progress'"),
                            {'id': analysis.id}
                        )
                        db.session.commit()
                    except Exception:
                        logger.exception("Raw SQL fallback also failed for analysis %s", analysis.id)
            raise

    # ── Core fallback loop ───────────────────────────────────────────────────

    def _evaluate_with_fallback(self, all_keys, eval_payload, spheres,
                                on_status=None, cancel_check=None):
        """
        Try each key in order until one succeeds.
        Returns (ratings, comments, raw_sample, winning_key_idx).
        """
        def emit(stage, progress, **extra):
            if on_status:
                on_status(stage, progress, **extra)

        total = len(all_keys)
        last_error = None
        # Track providers that returned a definitive error (4xx non-429) so we
        # can fast-fail subsequent keys for the same provider instead of waiting.
        dead_providers: set = set()

        for idx, (api_key, provider, config) in enumerate(all_keys):
            if cancel_check and cancel_check():
                raise RuntimeError("Aborted during key fallback.")

            model = config.get('model', '')

            # Helper: label for the NEXT key, used in failure messages.
            def _next_label(cur_idx=idx):
                if cur_idx + 1 < total:
                    _, np, nc = all_keys[cur_idx + 1]
                    return f'{np} ({nc.get("model", "")})'
                return None

            # Skip providers already confirmed dead (misconfigured key) unless
            # there is no other option left.
            remaining_alive = [
                i for i in range(idx, total)
                if all_keys[i][1] not in dead_providers
            ]
            if provider in dead_providers and len(remaining_alive) > 1:
                emit(
                    f'Skipping {provider} [{idx + 1}/{total}] — previously rejected',
                    20 + int(idx * 40 / max(total, 1)),
                    current_provider=provider,
                )
                continue

            emit(
                f'[{idx + 1}/{total}] Sending to {provider} ({model})',
                20 + int(idx * 40 / max(total, 1)),
                current_provider=provider,
                current_model=model,
            )

            try:
                response, request_body, endpoint_url = self._dispatch_request(
                    api_key, provider, config, eval_payload
                )

                if not response.ok:
                    status_code = response.status_code
                    # Read full error body for the log (cap at 500 chars)
                    error_body = response.text[:500].strip()
                    logger.warning("[%s/%s] %s HTTP %s: %s",
                                   idx + 1, total, provider, status_code, error_body)

                    # 4xx that aren't rate-limits mean the key/endpoint is wrong —
                    # mark provider dead so later keys for it are skipped.
                    if 400 <= status_code < 500 and status_code != 429:
                        dead_providers.add(provider)

                    next_lbl = _next_label()
                    emit(
                        f'[{idx + 1}/{total}] {provider} HTTP {status_code}: {error_body}'
                        + (f' → trying {next_lbl}' if next_lbl else ' → no more keys'),
                        20 + int(idx * 40 / max(total, 1)),
                        current_provider=provider,
                    )
                    response.raise_for_status()

                emit(
                    f'[{idx + 1}/{total}] {provider} responded — parsing',
                    65,
                    current_provider=provider,
                    current_model=model,
                )

                raw_resp = response.json()
                ratings, comments = self._parse_full_response(raw_resp, spheres, provider)

                # Completeness check — a sphere with all-NA scores means the LLM
                # only filled the first sphere and left the rest null (truncated
                # or didn't follow instructions).  Try the next key.
                incomplete = [
                    s.label for s in spheres
                    if not any(
                        ratings.get(str(q.id)) not in (None, 'NA')
                        for q in s.questions
                    )
                ]
                if incomplete:
                    next_lbl = _next_label()
                    raise RuntimeError(
                        f'{provider} incomplete — spheres {incomplete} have all-NA scores '
                        f'(response truncated or model non-compliant)'
                        + (f' → trying {next_lbl}' if next_lbl else '')
                    )

                raw_sample = {
                    'provider': provider,
                    'model': model,
                    'endpoint': endpoint_url,
                    'request': {k: v for k, v in request_body.items() if k != 'messages'}
                               if isinstance(request_body, dict) else None,
                    'response': {k: v for k, v in raw_resp.items()
                                 if k not in ('choices', 'content', 'candidates')}
                                if isinstance(raw_resp, dict) else None,
                }
                emit(
                    f'[{idx + 1}/{total}] {provider} succeeded',
                    80, current_provider=provider, current_model=model,
                )
                return ratings, comments, raw_sample, idx

            except Exception as e:
                last_error = e
                err_str = str(e)
                logger.warning("[%s/%s] %s failed: %s", idx + 1, total, provider, err_str)
                next_lbl = _next_label()
                # Only emit a "failed" message if the error wasn't already emitted
                # above (i.e. not an HTTP error which already logged inline).
                import requests as _req
                if not isinstance(e, _req.exceptions.HTTPError):
                    emit(
                        f'[{idx + 1}/{total}] {provider} failed: {err_str}'
                        + (f' → trying {next_lbl}' if next_lbl else ' → no more keys'),
                        20 + int(idx * 40 / max(total, 1)),
                        current_provider=provider,
                    )
                elif next_lbl:
                    emit(
                        f'→ trying {next_lbl}',
                        20 + int(idx * 40 / max(total, 1)),
                    )
                continue

        raise RuntimeError(
            f"All {total} key(s) exhausted. Last error: {last_error}"
        )

    # ── Prompt construction ──────────────────────────────────────────────────

    def _build_sphere_payload(self, country, sphere_list, additional_instructions=None):
        """Build the LLM prompt payload for one or more spheres."""
        if not isinstance(sphere_list, list):
            sphere_list = [sphere_list]  # backward-compat single sphere

        sphere_labels = ', '.join(s.label for s in sphere_list)
        payload = {
            "context": {
                "role": "Senior Institutional Analyst and Anti-Corruption Expert",
                "task": (
                    f"Evaluate the institutional legitimacy of '{country}' by scoring "
                    f"every question in ALL of the following spheres: {sphere_labels}."
                ),
                "scoring_scale": {
                    "1": "Extremely Weak / Highly Corrupt / Failed Institution",
                    "4": "Neutral / Average Performance",
                    "7": "Extremely Strong / Transparent / High Integrity",
                },
                "output_requirements": [
                    "Set 'score' to an integer 1-7 for every question in every sphere",
                    "Set 'reasoning' to 1-2 evidence-based sentences per question",
                    "Return the EXACT same JSON structure with ALL scores and reasoning filled in",
                    "Output valid JSON only — no markdown, no text outside the JSON",
                ],
            },
            "evaluation_target": country,
            "spheres": {
                s.name: {
                    "label": s.label,
                    "questions": {
                        str(q.id): {"content": q.content, "score": None, "reasoning": None}
                        for q in s.questions
                    },
                }
                for s in sphere_list
            },
        }
        if additional_instructions:
            payload["context"]["additional_instructions"] = additional_instructions
        return payload

    # ── HTTP dispatch ────────────────────────────────────────────────────────

    def _dispatch_request(self, key, provider, config, eval_payload):
        kind = config['kind']
        headers = {"Content-Type": "application/json"}
        user_content = json.dumps(eval_payload, ensure_ascii=False)
        system_instruction = (
            "You are a professional institutional analyst. "
            "Fill in every null 'score' (integer 1-7) and 'reasoning' (1-2 concise sentences) "
            "field in the JSON task, then return the complete JSON. "
            "Output valid JSON only — no markdown, no commentary."
        )

        if kind == 'openai_compatible':
            headers["Authorization"] = f"Bearer {key}"
            payload = {
                "model": config['model'],
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": config['max_tokens'],
                "response_format": {"type": "json_object"},
            }
            url = config['url']
            resp = requests.post(url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT)
            return resp, payload, url

        elif kind == 'anthropic':
            headers["x-api-key"] = key
            headers["anthropic-version"] = config['anthropic_version']
            payload = {
                "model": config['model'],
                "max_tokens": config['max_tokens'],
                "system": system_instruction,
                "messages": [{"role": "user", "content": user_content}],
                "temperature": 0.2,
            }
            url = config['url']
            resp = requests.post(url, headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT)
            return resp, payload, url

        elif kind == 'gemini':
            url = config['url'].format(model=config['model'])
            payload = {
                "system_instruction": {"parts": [{"text": system_instruction}]},
                "contents": [{"parts": [{"text": user_content}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                    "maxOutputTokens": config['max_tokens'],
                },
            }
            resp = requests.post(f"{url}?key={key}", headers=headers, json=payload, timeout=self.REQUEST_TIMEOUT)
            return resp, payload, url

        raise ValueError(f"Unsupported provider kind: {kind}")

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_full_response(self, data, spheres, provider):
        try:
            if provider in ('groq', 'openai', 'openrouter'):
                text = data['choices'][0]['message']['content']
            elif provider == 'claude':
                text_blocks = [b['text'] for b in data.get('content', []) if b.get('type') == 'text']
                if not text_blocks:
                    raise ValueError("No text content block in Claude response")
                text = text_blocks[0]
            elif provider == 'gemini':
                text = data['candidates'][0]['content']['parts'][0]['text']
            else:
                raise ValueError(f"No parser for provider: {provider}")

            text = text.strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start = text.find('{')
                if start == -1:
                    raise json.JSONDecodeError("No JSON found", text, 0)
                parsed, _ = json.JSONDecoder().raw_decode(text[start:])

            spheres_data = parsed.get('spheres', {})
            ratings, comments = {}, {}

            for sphere in spheres:
                sphere_resp = spheres_data.get(sphere.name, {})
                questions_resp = sphere_resp.get('questions', {})
                for q in sphere.questions:
                    qid = str(q.id)
                    item = questions_resp.get(qid, {})
                    raw_score = item.get('score')
                    try:
                        ratings[qid] = max(1, min(7, int(raw_score)))
                    except (ValueError, TypeError):
                        ratings[qid] = 'NA'
                    comments[qid] = item.get('reasoning') or 'AI reasoning not provided.'

            return ratings, comments

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error("Failed to parse AI response from %s: %s", provider, e)
            raise RuntimeError(f"Could not parse JSON response from {provider}: {e}")

    # ── Utilities ────────────────────────────────────────────────────────────

    def _fresh_get(self, analysis_id):
        """Load a fresh copy from DB, bypassing the identity-map cache.

        Uses rollback() (not commit()) to end any open transaction so the next
        read starts a new one with a fresh snapshot.  rollback() is safe even
        when there is nothing pending — it never tries to flush dirty state,
        so it cannot throw the way commit() can when the session is degraded.
        """
        try:
            db.session.rollback()
        except Exception:
            pass
        return AIAnalysis.get_by_id(analysis_id)

    def _update_status(self, aid, status, metadata, write_lock=None):
        """Write progress metadata without touching scores/comments."""
        with _optional_lock(write_lock):
            analysis = self._fresh_get(aid)
            if analysis is None:
                return
            analysis.status = status
            current = dict(analysis.metadata_json or {})
            # Clear transient per-attempt fields so stale data doesn't bleed across stages
            current.pop('provider_error', None)
            current.update(metadata)
            analysis.metadata_json = current
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(analysis, 'metadata_json')
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                logger.warning("Status update commit failed for analysis %s: %s", aid, e)

    def _calculate_normalized_sphere_avg(self, ratings):
        try:
            vals = [int(v) for v in ratings.values() if str(v).isdigit()]
            return round(sum(vals) / (len(vals) * 7), 3) if vals else 0.0
        except Exception:
            return 0.0

    def _resolve_all_api_keys(self, user_id, selected_key_id=None):
        from models.api_key_models import APIKey

        if selected_key_id == 'system':
            if self.env_api_key:
                return [(self.env_api_key, 'groq', self.PROVIDER_CONFIG['groq'])]
            return []

        user_keys = APIKey.get_active_user_keys(user_id)

        if selected_key_id:
            try:
                target_id = int(selected_key_id)
                selected = next((k for k in user_keys if k.id == target_id), None)
                if selected:
                    config = self.PROVIDER_CONFIG.get(selected.provider)
                    if config:
                        return [(selected.get_key(), selected.provider, config)]
            except (ValueError, TypeError):
                pass
            return []

        result = []
        for uk in user_keys:
            config = self.PROVIDER_CONFIG.get(uk.provider)
            if config:
                result.append((uk.get_key(), uk.provider, config))
        if self.env_api_key:
            result.append((self.env_api_key, 'groq', self.PROVIDER_CONFIG['groq']))
        return result


# ── Helpers ──────────────────────────────────────────────────────────────────

class _QSnap:
    """Immutable snapshot of a Question — independent of the SQLAlchemy session."""
    __slots__ = ('id', 'content')

    def __init__(self, id, content):
        self.id = id
        self.content = content


class _SphereSnap:
    """Immutable snapshot of a Sphere + its questions."""
    __slots__ = ('name', 'label', 'questions')

    def __init__(self, name, label, questions):
        self.name = name
        self.label = label
        self.questions = questions  # list of _QSnap


class _optional_lock:
    """Context manager that acquires a threading.Lock if provided, else is a no-op."""
    def __init__(self, lock):
        self._lock = lock

    def __enter__(self):
        if self._lock:
            self._lock.acquire()
        return self

    def __exit__(self, *_):
        if self._lock:
            self._lock.release()
