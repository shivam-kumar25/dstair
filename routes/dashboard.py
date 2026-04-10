import logging
from flask import Blueprint, render_template, redirect, url_for, abort
from flask_login import login_required, current_user
from services.analysis_service import AnalysisService

# Initialize Blueprint for the standard user dashboard
dashboard_bp = Blueprint('dashboard', __name__)
analysis_service = AnalysisService()
logger = logging.getLogger(__name__)

@dashboard_bp.route('/regular_user/<string:username>/dashboard')
@login_required
def index(username):
    """
    User Dashboard Page (HTML).
    Central hub for standard users to manage analyses and view global metrics.
    Redirects administrative or AI entities to their respective dedicated spaces.
    """
    try:
        # Role-based redirection using User model helper properties
        if current_user.is_admin:
            return redirect(url_for('admin.dashboard'))
        if current_user.is_ai:
            return redirect(url_for('ai_dashboard.index'))

        if current_user.user_account_unique_username_string != username:
            abort(403)

        user_id = current_user.unique_database_identifier_integer

        # Fetch user-specific data and aggregate metrics
        analyses = analysis_service.get_analyses_for_user(user_id)
        total_runs = analysis_service.count_analyses(user_id)
        unique_countries = analysis_service.count_unique_countries(user_id)
        triggered_tools_count = analysis_service.get_aggregated_triggered_tools_count(user_id)

        from models.core_models import Country
        countries = Country.get_all_ordered()

        return render_template('user/dashboard.html',
                               user=current_user,
                               analyses=analyses,
                               total_runs=total_runs,
                               unique_countries=unique_countries,
                               triggered_tools_count=triggered_tools_count,
                               countries=countries)

    except Exception as e:
        logger.exception("Unexpected error rendering dashboard for user %s", current_user.user_account_unique_username_string)
        raise e

@dashboard_bp.route('/regular_user/<username>/tools')
@login_required
def tools(username):
    """
    User Tools Library Page (HTML).
    Enumerates all anti-corruption tools, prioritizing those triggered in past analyses.
    """
    try:
        if current_user.user_account_unique_username_string != username:
            abort(403)

        user_id = current_user.unique_database_identifier_integer

        # Identify tools that have been unlocked by any of the user's analyses
        triggered_tools = analysis_service.get_aggregated_user_tools(user_id)
        triggered_tools_ids = [t.id for t in triggered_tools]
        all_tools = analysis_service.get_all_tools()

        # Sort: Triggered (unlocked) tools first, then by ID
        sorted_tools = sorted(all_tools, key=lambda t: (0 if t.id in triggered_tools_ids else 1, t.id))

        return render_template('user/dashboard_tools.html',
                               tools=sorted_tools,
                               triggered_tools_ids=triggered_tools_ids)
    except Exception as e:
        logger.exception("Unexpected error rendering tools library for user %s", current_user.user_account_unique_username_string)
        raise e
