import atexit
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Blueprint, render_template, request, jsonify, current_app, redirect, url_for
from flask_login import current_user
from utils.decorators import ai_role_required_decorator
from services.ai_service import AIService
from services.api_key_service import APIKeyService
from models import APIKey, AIAnalysis, Country
from extensions import limiter

ai_dashboard_bp = Blueprint('ai_dashboard', __name__, url_prefix='/ai')

# Module-level singletons — both are stateless and safe to share across threads.
ai_service = AIService()
api_key_service = APIKeyService()

EVAL_TIMEOUT_SECONDS = 900  # 15 minutes — allows 9 spheres × 60s + key fallback headroom
logger = logging.getLogger(__name__)

# Background pool for long-running LLM evaluation tasks (workflow §9 step 3)
ai_executor = ThreadPoolExecutor(max_workers=4)
# Shut the pool down gracefully when the process exits so threads are not abandoned.
atexit.register(lambda: ai_executor.shutdown(wait=False))

# ── Per-country submission lock ──────────────────────────────────────────────
# Prevents two simultaneous HTTP requests from both passing the in_progress check
# and submitting duplicate background evaluations for the same country.
# Scoped to this process; safe with Waitress threads (shared memory).
_country_eval_locks: dict[str, threading.Lock] = {}
_country_eval_locks_mutex = threading.Lock()


def _get_country_lock(country_code: str) -> threading.Lock:
    with _country_eval_locks_mutex:
        if country_code not in _country_eval_locks:
            _country_eval_locks[country_code] = threading.Lock()
        return _country_eval_locks[country_code]


# ── Per-analysis write lock ──────────────────────────────────────────────────
# Serialises metadata_json writes from the background evaluation thread and the
# timeout callback so they don't clobber each other.
_analysis_write_locks: dict[int, threading.Lock] = {}
_analysis_write_locks_mutex = threading.Lock()

# IDs of analyses that have been cancelled by the user
_cancelled_analyses: set[int] = set()
_cancelled_analyses_lock = threading.Lock()


def _get_analysis_lock(analysis_id: int) -> threading.Lock:
    with _analysis_write_locks_mutex:
        if analysis_id not in _analysis_write_locks:
            _analysis_write_locks[analysis_id] = threading.Lock()
        return _analysis_write_locks[analysis_id]


def is_cancelled(analysis_id: int) -> bool:
    with _cancelled_analyses_lock:
        return analysis_id in _cancelled_analyses


# ── Pages ───────────────────────────────────────────────────────────────────

@ai_dashboard_bp.route('/dashboard')
@ai_role_required_decorator
def index():
    """AI Dashboard — stats and archive of all country evaluations (workflow §4 AI)."""
    # Reset any records stuck in_progress from a previous crash/unclean shutdown.
    # Runs on dashboard load so stuck records are always visible as errors here.
    stale_timeout_minutes = max(1, EVAL_TIMEOUT_SECONDS // 60 + 1)
    AIAnalysis.reset_stale_in_progress(older_than_minutes=stale_timeout_minutes)

    # Use summary query — skips large score/comment JSON columns not needed for list view.
    all_ai_evals = AIAnalysis.get_all_summary()
    all_ai_evals.sort(key=lambda x: x.updated_at or x.created_at, reverse=True)

    return render_template(
        'ai/dashboard.html',
        user=current_user,
        analyses=all_ai_evals,
        total_runs=len(all_ai_evals),
        completed_runs=sum(1 for a in all_ai_evals if a.status == 'completed'),
        in_progress_runs=sum(1 for a in all_ai_evals if a.status == 'in_progress')
    )


@ai_dashboard_bp.route('/analysis')
@ai_role_required_decorator
def analysis():
    """Country evaluation trigger page (workflow §9 step 1)."""
    # Summary query is enough here — we only need country codes and statuses.
    all_ai_evals = AIAnalysis.get_all_summary()
    all_ai_evals.sort(key=lambda x: x.updated_at or x.created_at, reverse=True)
    evaluated_countries = {a.country for a in all_ai_evals if a.status == 'completed'}
    all_countries = Country.get_all_ordered()

    active_keys = APIKey.get_active_user_keys(current_user.unique_database_identifier_integer)
    active_keys_data = [k.to_dict() for k in active_keys]
    has_system_key = bool(os.getenv('GROQ_API_KEY'))

    return render_template(
        'ai/analysis.html',
        user=current_user,
        analyses=all_ai_evals,
        all_countries=all_countries,
        evaluated_countries=evaluated_countries,
        active_keys=active_keys_data,
        has_system_key=has_system_key,
    )


@ai_dashboard_bp.route('/analysis/<int:analysis_id>')
@ai_role_required_decorator
def view_analysis(analysis_id):
    """View an AI analysis result (any status)."""
    ai_analysis = AIAnalysis.get_by_id(analysis_id)
    if not ai_analysis:
        return redirect(url_for('ai_dashboard.index'))
    from models.core_models import Sphere
    spheres = Sphere.get_all_ordered()
    questions_map = {str(q.id): q for s in spheres for q in s.questions}
    return render_template(
        'ai/analysis_view.html',
        user=current_user,
        ai_analysis=ai_analysis,
        spheres=spheres,
        questions_map=questions_map
    )


@ai_dashboard_bp.route('/api-keys')
@ai_role_required_decorator
def api_keys():
    """BYOK API key manager (workflow §8)."""
    user_keys = api_key_service.get_user_keys(current_user.unique_database_identifier_integer)
    keys_by_provider = {}
    for key in user_keys:
        keys_by_provider.setdefault(key.provider, []).append(key.to_dict())

    return render_template(
        'ai/api_keys.html',
        user=current_user,
        providers=APIKey.PROVIDERS,
        keys_by_provider=keys_by_provider
    )


# ── AI Evaluation API ────────────────────────────────────────────────────────

@ai_dashboard_bp.route('/analysis/evaluate', methods=['POST'])
@ai_role_required_decorator
@limiter.limit("5 per minute")
def evaluate():
    """
    Trigger a background AI country evaluation (workflow §9 steps 2–3).
    Returns immediately with the analysis ID. Frontend polls /analysis/<id>/status.
    Per spec: if a record exists for the country, reset in-place; else create new.
    """
    from utils.sanitizer import sanitize_input
    data = request.get_json(silent=True) or {}
    country_code = (data.get('country') or '').strip()
    raw_instructions = sanitize_input(data.get('additional_instructions', '') or '')
    additional_instructions = raw_instructions.strip()[:1000] or None  # strip HTML, cap at 1000 chars
    selected_key_id = data.get('selected_key_id') or None  # int key ID, 'system', or None

    if not country_code:
        return jsonify({'success': False, 'error': 'Country code is required.'}), 400

    country_record = Country.find_one(code=country_code) or Country.find_one(name=country_code)
    if not country_record:
        return jsonify({'success': False, 'error': f'Country "{country_code}" not recognized.'}), 400

    country_code = country_record.code

    try:
        # ── Stale record cleanup ─────────────────────────────────────────────
        # Reset any in_progress records that are older than the eval timeout —
        # they were abandoned by a previous crash or unclean shutdown.
        stale_timeout_minutes = max(1, EVAL_TIMEOUT_SECONDS // 60 + 1)
        AIAnalysis.reset_stale_in_progress(older_than_minutes=stale_timeout_minutes)

        # ── Per-country lock ─────────────────────────────────────────────────
        # Prevents two simultaneous requests from both passing the in_progress
        # check and submitting duplicate background tasks for the same country.
        country_lock = _get_country_lock(country_code)
        if not country_lock.acquire(blocking=False):
            return jsonify({
                'success': False,
                'error': 'An evaluation is already being submitted for this country.'
            }), 409

        try:
            ai_analysis = AIAnalysis.get_by_country(country_code)

            if ai_analysis and ai_analysis.status == 'in_progress':
                return jsonify({
                    'success': False,
                    'error': 'An evaluation is already in progress for this country.'
                }), 409

            # Create new record or reset existing one — single save via mark_in_progress.
            if not ai_analysis:
                ai_analysis = AIAnalysis(country=country_code)
            ai_analysis.mark_in_progress()
            analysis_id = ai_analysis.id
        finally:
            # Release as soon as the record is committed so other requests see
            # the updated status and return 409 via the DB check above.
            country_lock.release()

        app = current_app._get_current_object()
        user_id = current_user.unique_database_identifier_integer
        write_lock = _get_analysis_lock(analysis_id)

        def run_eval(app_ctx, uid, code, aid, instr, key_id):
            with app_ctx.app_context():
                try:
                    ai_service.evaluate_country(uid, code, existing_analysis_id=aid,
                                                additional_instructions=instr,
                                                selected_key_id=key_id,
                                                write_lock=write_lock,
                                                cancel_check=lambda: is_cancelled(aid))
                except Exception as e:
                    logger.error("Background AI evaluation failed for %s: %s", code, e)
                    # ai_service already tried to mark_error; this is a safety net.
                    # If the ORM session is broken, fall back to raw SQL so the record
                    # is never left stuck in in_progress indefinitely.
                    with write_lock:
                        from extensions import db as _db
                        try:
                            _db.session.rollback()
                            record = AIAnalysis.get_by_id(aid)
                            if record and record.status == 'in_progress':
                                record.mark_error(str(e))
                        except Exception as inner:
                            logger.error("run_eval ORM fallback failed for analysis %s: %s", aid, inner)
                            try:
                                _db.session.rollback()
                                from sqlalchemy import text
                                _db.session.execute(
                                    text("UPDATE ai_analyses SET status='error', "
                                         "updated_at=CURRENT_TIMESTAMP "
                                         "WHERE id=:id AND status='in_progress'"),
                                    {'id': aid}
                                )
                                _db.session.commit()
                            except Exception:
                                logger.exception("Raw SQL fallback failed for analysis %s — record may be stuck", aid)

        future = ai_executor.submit(run_eval, app, user_id, country_code, analysis_id,
                                    additional_instructions, selected_key_id)

        def _on_timeout():
            with app.app_context():
                with write_lock:
                    record = AIAnalysis.get_by_id(analysis_id)
                    if record and record.status == 'in_progress':
                        record.mark_error(f"Evaluation timed out after {EVAL_TIMEOUT_SECONDS}s.")
                        logger.warning("AI eval timeout: analysis_id=%s country=%s",
                                       analysis_id, country_code)

        timeout_timer = threading.Timer(EVAL_TIMEOUT_SECONDS, _on_timeout)
        timeout_timer.daemon = True
        timeout_timer.start()
        future.add_done_callback(lambda _: timeout_timer.cancel())

        return jsonify({'success': True, 'analysis_id': analysis_id})

    except Exception:
        logger.exception("Failed to start AI evaluation for %s", country_code)
        return jsonify({'success': False, 'error': 'Failed to start evaluation.'}), 500


@ai_dashboard_bp.route('/analysis/<int:analysis_id>/status')
@ai_role_required_decorator
def analysis_status(analysis_id):
    """Poll endpoint for frontend during evaluation (workflow §9 step 10)."""
    ai_analysis = AIAnalysis.get_by_id(analysis_id)
    if not ai_analysis:
        return jsonify({'success': False, 'error': 'Not found.'}), 404

    return jsonify({
        'success': True,
        'id': ai_analysis.id,
        'country': ai_analysis.country,
        'status': ai_analysis.status,
        'metadata': ai_analysis.metadata_json or {},
        'completed': ai_analysis.status == 'completed',
        'error': ai_analysis.status == 'error'
    })


@ai_dashboard_bp.route('/analysis/<int:analysis_id>/cancel', methods=['POST'])
@ai_role_required_decorator
def cancel_analysis(analysis_id):
    """Cancel a running evaluation."""
    ai_analysis = AIAnalysis.get_by_id(analysis_id)
    if not ai_analysis:
        return jsonify({'success': False, 'error': 'Not found.'}), 404
    if ai_analysis.status != 'in_progress':
        return jsonify({'success': False, 'error': 'Not in progress.'}), 400

    with _cancelled_analyses_lock:
        _cancelled_analyses.add(analysis_id)

    write_lock = _get_analysis_lock(analysis_id)
    with write_lock:
        ai_analysis.mark_error('Cancelled by user.')
    return jsonify({'success': True, 'message': 'Evaluation cancelled.'})


@ai_dashboard_bp.route('/analysis/<int:analysis_id>/delete', methods=['DELETE'])
@ai_role_required_decorator
def delete_analysis(analysis_id):
    """Permanently delete an AI analysis record (workflow §4 AI, archive)."""
    ai_analysis = AIAnalysis.get_by_id(analysis_id)
    if not ai_analysis:
        return jsonify({'success': False, 'error': 'Not found.'}), 404

    try:
        ai_analysis.delete()
        return jsonify({'success': True, 'message': 'Analysis deleted.'})
    except Exception:
        logger.exception("Error deleting AI analysis %s", analysis_id)
        return jsonify({'success': False, 'error': 'Failed to delete.'}), 500


# ── API Key Management ───────────────────────────────────────────────────────

@ai_dashboard_bp.route('/api-keys/save', methods=['POST'])
@ai_role_required_decorator
def save_api_key():
    """Encrypt and persist a new (or updated) provider API key (workflow §8a)."""
    data = request.get_json(silent=True) or {}
    provider = data.get('provider', '').strip()
    api_key_value = data.get('api_key', '').strip()

    if not provider or provider not in APIKey.PROVIDERS:
        return jsonify({'success': False, 'error': 'Invalid provider.'}), 400
    if not api_key_value:
        return jsonify({'success': False, 'error': 'API key cannot be empty.'}), 400

    # key_id present → edit existing record; absent → create new
    raw_key_id = data.get('key_id') or None
    key_id = None
    if raw_key_id is not None:
        try:
            key_id = int(raw_key_id)
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid key ID.'}), 400

    try:
        api_key_service.save_key(
            current_user.unique_database_identifier_integer,
            provider,
            api_key_value,
            key_id=key_id,
        )
        msg = 'API key updated.' if key_id else 'API key saved.'
        return jsonify({'success': True, 'message': msg})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception:
        logger.exception("Failed to save API key for provider %s", provider)
        return jsonify({'success': False, 'error': 'Failed to save key.'}), 500


@ai_dashboard_bp.route('/api-keys/<int:key_id>/toggle', methods=['POST'])
@ai_role_required_decorator
def toggle_api_key(key_id):
    """Toggle a key active/inactive (workflow §8b)."""
    try:
        is_active = api_key_service.toggle_key(
            current_user.unique_database_identifier_integer, key_id
        )
        msg = 'Key enabled.' if is_active else 'Key disabled.'
        return jsonify({'success': True, 'is_active': is_active, 'message': msg})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception:
        logger.exception("Error toggling API key %s", key_id)
        return jsonify({'success': False, 'error': 'Failed to toggle key.'}), 500


@ai_dashboard_bp.route('/api-keys/<int:key_id>/delete', methods=['DELETE'])
@ai_role_required_decorator
def delete_api_key(key_id):
    """Delete a key (workflow §8d)."""
    try:
        api_key_service.delete_key(
            current_user.unique_database_identifier_integer, key_id
        )
        return jsonify({'success': True, 'message': 'API key deleted.'})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception:
        logger.exception("Error deleting API key %s", key_id)
        return jsonify({'success': False, 'error': 'Failed to delete key.'}), 500


@ai_dashboard_bp.route('/api-keys/reorder', methods=['POST'])
@ai_role_required_decorator
def reorder_api_keys():
    """Update key execution priority order (workflow §8c)."""
    data = request.get_json(silent=True) or {}
    key_order = data.get('order', [])

    if not key_order or not isinstance(key_order, list):
        return jsonify({'success': False, 'error': 'Invalid ordering data.'}), 400

    try:
        int_order = [int(k) for k in key_order]
        api_key_service.reorder_keys(current_user.unique_database_identifier_integer, int_order)
        return jsonify({'success': True, 'message': 'Key order updated.'})
    except Exception:
        logger.exception("Error reordering API keys")
        return jsonify({'success': False, 'error': 'Failed to reorder keys.'}), 500
