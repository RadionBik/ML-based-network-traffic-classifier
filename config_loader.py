import configparser

class NoConfig(Exception):
    pass

class Config_Init:
    """ DEPRECATED """
    def __init__(self, config_file='config.ini'):
        """
        inits parameters from the config file, loads csv. into a DataFrame
        """
        self._config = configparser.ConfigParser()
        read_files = self._config.read(config_file)
        if not read_files:
            raise NoConfig(
                'Config {} not found. Please check it exists'.format(
                    config_file
                )
            )


    def get(self):
        return self._config
