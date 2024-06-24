def setup_logging(name, level = 'INFO'):
    try:
        import logging
        import logging.config
        import coloredlogs
        logger = logging.getLogger(name)
        if logger.hasHandlers():
            logger.handlers.clear()
        coloredlogs.install(
            fmt = '[%(name)s] %(levelname)s: %(message)s',
            level = level.upper(),  # suppress redundant logs if level is not 'INFO'
            level_styles = {
                'warning': {'color': 'yellow'},
                'error': {'color': 'red'},
                'critical': {'color': 'red', 'bold': True}
            }
        )  
        return logger
    except Exception as e:
        print(e)
        return None