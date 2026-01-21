import time
import threading
import logging

logger = logging.getLogger(__name__)


def monitor_idle_models(model_manager, interval, timeout):
    """
    Background task that periodically checks for idle models
    and unloads them if idle time exceeds timeout.
    """
    logger.info(f"Idle model monitor started (check every {interval}s, timeout {timeout}s)")

    while True:
        try:
            time.sleep(interval)
            unloaded = model_manager.unload_idle_models(timeout)
            if unloaded:
                logger.info(f"Auto-unloaded models: {', '.join(m for m in unloaded if m)}")
        except Exception as e:
            logger.error(f"Error in idle model monitor: {e}", exc_info=True)


def start_idle_monitor(model_manager, interval, timeout):
    """Start the idle model monitor as a daemon thread."""
    thread = threading.Thread(
        target=monitor_idle_models,
        args=(model_manager, interval, timeout),
        daemon=True,
        name="IdleModelMonitor"
    )
    thread.start()
    logger.info("Idle model monitor thread started")
    return thread
