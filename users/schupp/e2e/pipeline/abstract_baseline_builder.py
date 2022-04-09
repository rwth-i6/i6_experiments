# Python class to create arbitrary baselines

# Idea is, this should be used to define a abitrary pipeline:
# e.g.: Switchboard pipeline

class AbstractPipeline:

    def get_train_model():
        return None
    
    def get_recog_model():
        return None

class SWBPipeline(AbstractPipeline):
    '''
1) First create a return config 
2) Start some training
    '''
    def main():
        config_train = self.get_train_config()
        _present_w_type(config_train)

        # write config job

        train_job = self.get_train_job(config_train)
        _present_w_type(config_train)


    def _present_w_type(var, _type=None):
        assert(var != None, "value empty")
        if _type != None:
            assert(isinstance(var, _type), "wrong type")


class SHeadAttentionPipeline(SWBPipeline):
    def get_train_config():
        return ReturnnConfig(config=config_params, python_prolog=pre_python_code)

    def get_train_job(config_train):
        return ReturnnTrainingJob(config, log_verbosity=5, num_epochs=200, **train_reciepe_defaults)