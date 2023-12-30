
import pickle
from pathlib import Path
import os, sys
import yaml


class Config :

    def __init__(self, config_file) :
        self.file = config_file

    @property 
    def project_path(self) :
        return self.file['project_path']

    def model_path(self, model_name) :
        return self.file['model_path'][model_name]
    
    def template(self, template_type) :
        return self.file['template'][template_type]

    @property 
    def device(self) :
        return self.file['device']



def load_config() :
    current_file_path = Path(__file__).absolute()
    project_path = current_file_path.parents[1]
    config_path = project_path.joinpath("config/config.yaml")

    with open(config_path) as f : 
        config = yaml.load(f, yaml.SafeLoader)

    return Config(config)




