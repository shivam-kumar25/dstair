from extensions import db
from datetime import datetime, timezone, timedelta
from models.base import ActiveRecordMixin
from sqlalchemy.orm.attributes import flag_modified

class AIAnalysis(ActiveRecordMixin, db.Model):
    """
    Stores AI-generated evaluations for sovereign countries.
    Exactly one record per country — enforced by the Unique constraint on country.
    Decoupled from User and Analysis models — AI evaluations are global, not user-scoped.

    Status lifecycle: not_started → in_progress → completed
                                                 ↘ error
    """
    __tablename__ = 'ai_analyses'

    id = db.Column(db.Integer, primary_key=True)

    # FK to Country.code — one record per country enforced here.
    country = db.Column(db.String(100), db.ForeignKey('countries.code', onupdate='CASCADE'), unique=True, nullable=False, index=True)

    # Lifecycle state
    status = db.Column(db.String(20), default='not_started', index=True)

    # Structure: {"question_id": score_value}
    # Scores are on the 1–7 scale. Null until evaluation completes.
    ai_scores_for_all_questions = db.Column(db.JSON, default=dict)

    # Structure: {"question_id": "reasoning text"}
    # AI-generated rationale per question. Null until evaluation completes.
    ai_comments_for_all_questions = db.Column(db.JSON, default=dict)

    # Generation metadata: provider used, model version, generation timestamp, retries, etc.
    metadata_json = db.Column(db.JSON, default=dict)

    # created_at: when the record was first created (seed time or first evaluation trigger)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    # updated_at: updated when evaluation completes or fails
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<AIAnalysis {self.country}: {self.status}>'

    @property
    def title(self):
        """Dynamic title for the UI archive list."""
        return f"AI Evaluation: {self.country}"

    @property
    def country_obj(self):
        """Lookup the Country model for this AI analysis.

        Uses a Flask request-local cache (flask.g) so that when a page renders
        a list of analyses the Country table is hit only once (all countries are
        loaded into a dict on first access) rather than once per row (N+1).
        Falls back to a direct query outside of request context (CLI, tests).
        """
        from models.core_models import Country
        try:
            from flask import g
            if not hasattr(g, '_ai_country_cache'):
                g._ai_country_cache = {c.code: c for c in Country.get_all_ordered()}
            return g._ai_country_cache.get(self.country)
        except RuntimeError:
            # Outside request context — direct query is fine
            return Country.get_by_code(self.country)

    def to_dict(self):
        return {
            'id': self.id,
            'country': self.country,
            'title': self.title,
            'status': self.status,
            'ai_comments_for_all_questions': self.ai_comments_for_all_questions or {},
            'ai_scores_for_all_questions': self.ai_scores_for_all_questions or {},
            'metadata': self.metadata_json or {},
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M') if self.updated_at else None
        }

    @classmethod
    def get_all_summary(cls):
        """Load all records WITHOUT the large score/comment JSON columns.

        Use this for list/dashboard views that only need status, country, dates,
        and metadata aggregates — not the full per-question data.  Deferred columns
        are fetched lazily only if accessed, so the caller must not touch
        ai_scores_for_all_questions or ai_comments_for_all_questions on the
        returned objects.
        """
        from sqlalchemy.orm import defer as sa_defer
        return (
            db.session.query(cls)
            .options(
                sa_defer(cls.ai_scores_for_all_questions),
                sa_defer(cls.ai_comments_for_all_questions),
            )
            .all()
        )

    @classmethod
    def get_by_country(cls, country_code: str):
        return db.session.query(cls).filter_by(country=country_code).first()

    @classmethod
    def reset_stale_in_progress(cls, older_than_minutes: int = 10) -> int:
        """
        Reset any records stuck in 'in_progress' whose updated_at is older than
        *older_than_minutes*.  Returns the number of records reset.

        Called at the start of each evaluation trigger so users can always retry
        after a crash or unclean server shutdown.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
        stale = (
            db.session.query(cls)
            .filter(cls.status == 'in_progress', cls.updated_at < cutoff)
            .all()
        )
        for record in stale:
            record.status = 'error'
            if not record.metadata_json:
                record.metadata_json = {}
            record.metadata_json['last_error'] = (
                f'Evaluation interrupted — reset after {older_than_minutes} min stale timeout.'
            )
            flag_modified(record, 'metadata_json')
        if stale:
            db.session.commit()
        return len(stale)

    def mark_in_progress(self, commit: bool = True):
        """Reset to in_progress, clearing previous results per overwrite behavior spec."""
        self.status = 'in_progress'
        self.updated_at = datetime.now(timezone.utc)
        self.ai_scores_for_all_questions = None
        self.ai_comments_for_all_questions = None
        self.metadata_json = {}  # clear stale progress/stage from any previous run
        flag_modified(self, "ai_scores_for_all_questions")
        flag_modified(self, "ai_comments_for_all_questions")
        flag_modified(self, "metadata_json")
        if commit:
            self.save()

    def mark_completed(self, scores: dict, comments: dict, metadata: dict = None, commit: bool = True):
        # Don't overwrite a timeout-error that fired before we could finish.
        # The completed data is still valid — we always prefer it over an error state.
        self.status = 'completed'
        self.updated_at = datetime.now(timezone.utc)
        self.ai_scores_for_all_questions = scores
        self.ai_comments_for_all_questions = comments
        if metadata is not None:
            self.metadata_json = metadata
        flag_modified(self, "ai_scores_for_all_questions")
        flag_modified(self, "ai_comments_for_all_questions")
        flag_modified(self, "metadata_json")
        if commit:
            self.save()

    def mark_error(self, error_msg: str, commit: bool = True):
        # Never overwrite a successful completion — if the task finished just before
        # the timeout callback fired, keep the completed state.
        if self.status == 'completed':
            return self
        self.status = 'error'
        self.updated_at = datetime.now(timezone.utc)
        if not self.metadata_json:
            self.metadata_json = {}
        self.metadata_json['last_error'] = error_msg
        flag_modified(self, "metadata_json")
        if commit:
            self.save()
        return self
