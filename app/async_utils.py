import asyncio
import platform

def get_event_loop():
    """Get the appropriate event loop based on platform"""
    if platform.system() == 'Windows':
        # Windows-specific setup (though we shouldn't need this on Render)
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()