import os
import logging

from flask import Flask, request
from config import DevelopmentConfig, ProductionConfig, validate_runtime_config
from extensions import db, migrate, csrf, limiter
from flask_login import LoginManager


def create_app(config_class=None):
    """
    Application factory pattern.
    Creates and configures the Flask application instance.
    This pattern allows creating multiple instances of the app
    (e.g., for testing or different environments) cleanly.
    """
    # ── Environment-Aware Config Selection ─────────────────────
    if config_class is None:
        env = os.getenv("FLASK_ENV", "development").lower()
        config_class = ProductionConfig if env == "production" else DevelopmentConfig

    # Define absolute path to the root directory for reliable template/static paths
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Initialize Flask app explicitly pointing to folders
    app = Flask(
        __name__,
        template_folder=os.path.join(root_path, "templates"),
        static_folder=os.path.join(root_path, "static"),
    )

    # Load configuration from the specified config class
    app.config.from_object(config_class)
    validate_runtime_config(app, config_class)

    # ── Logging Setup ─────────────────────────────────────────────
    # Configures standard Python logging to capture app events
    logging.basicConfig(
        level=app.config.get("LOG_LEVEL", logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── Flask Extensions Initialization ───────────────────────────
    # We initialize extensions here to bind them to this specific app instance
    db.init_app(app)  # SQLAlchemy ORM

    # ── SQLite Concurrency Fix ────────────────────────────────────
    # WAL mode allows concurrent reads during background writes (AI evaluation).
    # busy_timeout prevents "database is locked" errors by retrying for up to 20s.
    import sqlalchemy as _sa

    with app.app_context():
        @_sa.event.listens_for(db.engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):
            if "sqlite" in str(db.engine.url):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=20000")
                cur.close()

    migrate.init_app(
        app, db, render_as_batch=True
    )  # Alembic migrations (render_as_batch helps with SQLite table alters)
    csrf.init_app(app)  # CSRF protection (WTForms)
    limiter.init_app(app)  # Rate limiting

    # Initialize Flask-Login for user session management
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"  # Redirects unauthenticated requests to /login

    @login_manager.user_loader
    def load_user(user_id):
        """
        Callback used by Flask-Login to reload the user object from the user ID
        stored in the session.
        """
        from services.user_service import UserService

        return UserService().get_user_by_id(int(user_id))

    # ── Blueprints Registration ───────────────────────────────────
    # Blueprints segregate routes into logical modules (Separation of Concerns)
    from routes.public import public_bp
    from routes.onboarding import onboarding_bp
    from routes.auth import auth_bp
    from routes.admin import admin_bp
    from routes.analysis import analysis_bp
    from routes.dashboard import dashboard_bp
    from routes.ai_dashboard import ai_dashboard_bp

    # Register all domains
    app.register_blueprint(public_bp)
    app.register_blueprint(onboarding_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(ai_dashboard_bp)

    # ── CSRF Protection ──────────────────────────────────────────
    # All blueprints now have CSRF protection enabled.
    # Frontend AJAX/fetch calls include the token via X-CSRFToken header,
    # which Flask-WTF automatically validates.

    # ── Centralized Error Handlers ────────────────────────────────
    from core.error_handlers import register_error_handlers

    register_error_handlers(app)

    # ── Security Headers ─────────────────────────────────────────
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        # Content-Security-Policy — restricts script/style/image sources to
        # same-origin plus the CDNs actually used by the app (flagcdn, fonts).
        # Inline styles are allowed for legacy template compatibility; tighten
        # further once templates are audited for nonce support.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "img-src 'self' data: https://flagcdn.com https://images.pexels.com; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        if not app.debug:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response

    # ── Database Auto-Initialization & CLI ──────────────────────────
    # Registers the custom 'flask seed-db' command for one-step setup.
    @app.cli.command("seed-db")
    def seed_db_command():
        """Initialize the database and run seed scripts."""
        from utils.db_init import ensure_database_initialized

        with app.app_context():
            ensure_database_initialized(force_seed=True)
        print("Database initialized and seeded successfully.")

    # ── Graceful Degradation (Database Check) ─────────────────────
    # Ensures the app doesn't crash if the database is missing its tables.
    # Uses an in-memory flag to avoid checking the DB inspector on every request.
    _db_verified = {"initialized": False}

    @app.before_request
    def check_db_initialized():
        # Skip if already verified this process lifetime
        if _db_verified["initialized"]:
            return

        # Allow access to the maintenance page, static assets, and missing endpoints
        if request.endpoint is None or request.endpoint in ("maintenance", "static"):
            return

        import sqlalchemy

        try:
            inspector = sqlalchemy.inspect(db.engine)
            if not inspector.has_table("user"):
                if app.config.get("AUTO_INIT_DB", False):
                    from utils.db_init import ensure_database_initialized

                    ensure_database_initialized(force_seed=True)
                    _db_verified["initialized"] = True
                    return
                from flask import render_template

                return render_template("public/maintenance.html"), 503
            else:
                # Table exists — but we might need to run migration helpers (like adding columns)
                from utils.db_init import ensure_database_initialized

                ensure_database_initialized(force_seed=False)

            # Table exists & migrations checked — mark as verified
            _db_verified["initialized"] = True
        except Exception as e:
            app.logger.error(f"Database connection error: {e}")
            from flask import render_template

            return render_template("public/maintenance.html"), 503

    @app.route("/maintenance")
    def maintenance():
        from flask import render_template

        return render_template("public/maintenance.html")

    return app
