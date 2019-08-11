import configparser

class Config_Init:
    """ DEPRECATED """
    def __init__(self, config_file='config.ini'):
        """
        inits parameters from the config file, loads csv. into a DataFrame
        """
        self._config = configparser.ConfigParser()
        self._config.read(config_file)
        
    def get(self):
        return self._config