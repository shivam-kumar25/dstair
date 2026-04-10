import logging
from flask import jsonify, render_template, request
from core.exceptions import ApplicationSpecificBaseError, RequestedResourceNotFoundError

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    @app.errorhandler(ApplicationSpecificBaseError)
    def handle_application_specific_base_error(error):
        if request.is_json or request.path.startswith('/api/'):
            response = jsonify(error.convert_error_to_dictionary_representation())
            response.status_code = error.http_response_status_code_integer
            return response
        else:
            return render_template('public/500.html', error=error.exception_error_message_string), error.http_response_status_code_integer

    @app.errorhandler(400)
    def bad_request(e):
        if request.is_json or request.path.startswith('/api/') or request.path.startswith('/ai/'):
            return jsonify({'success': False, 'error': str(e.description) if hasattr(e, 'description') else 'Bad request'}), 400
        return render_template('public/400.html'), 400

    @app.errorhandler(403)
    def forbidden(e):
        if request.is_json or request.path.startswith('/api/') or request.path.startswith('/ai/'):
            return jsonify({'success': False, 'error': 'Forbidden'}), 403
        return render_template('public/403.html'), 403

    @app.errorhandler(404)
    def page_not_found(e):
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Not found'}), 404
        return render_template('public/404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.exception("Internal server error")
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
        return render_template('public/500.html', error="An unexpected error occurred."), 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(e):
        logger.exception("Unhandled exception: %s", e)
        if request.is_json or request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'An unexpected error occurred'}), 500
        return render_template('public/500.html', error="An unexpected error occurred."), 500
