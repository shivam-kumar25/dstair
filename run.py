"""
DSTAIR Application Launcher
============================
Smart entry point that uses:
  - waitress (production WSGI) when FLASK_ENV=production or by default
  - Flask dev server only when FLASK_ENV=development AND --debug flag

Usage:
    python run.py              → Production mode (waitress, no debug)
    python run.py --dev        → Development mode (Flask dev server, debug=True)
"""
import os
import sys
from app import create_app

app = create_app()

if __name__ == '__main__':
    is_dev = '--dev' in sys.argv or os.getenv('FLASK_ENV', '').lower() == 'development'
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')

    if is_dev:
        print(f"  🔧 Starting DSTAIR in DEVELOPMENT mode on http://127.0.0.1:{port}")
        import sys as _sys
        _python_lib = os.path.dirname(_sys.executable)
        app.run(
            debug=True,
            host='127.0.0.1',
            port=port,
            exclude_patterns=[
                os.path.join(_python_lib, '*'),
                os.path.join(_python_lib, 'Lib', '*'),
            ]
        )
    else:
        try:
            from waitress import serve
            print(f"  🚀 Starting DSTAIR with Waitress on http://{host}:{port}")
            print(f"     Press Ctrl+C to stop.")
            serve(app, host=host, port=port, threads=4)
        except ImportError:
            print("  ⚠️  waitress not installed. Falling back to Flask dev server.")
            print("     Install with: pip install waitress")
            app.run(debug=False, host=host, port=port)
