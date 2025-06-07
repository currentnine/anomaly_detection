
import torch
import constants as const
import fastflow

class SingletonModel:
    _instance = None
    
    @classmethod
    # def get_instance(cls):
    #     if cls._instance is None:
    #         cls._instance = cls._create_instance()
    #     return cls._instance
    def get_instance(cls):
        if cls._instance is None:
            print("Creating new model instance.")
            cls._instance = cls._create_instance()
            if cls._instance is None:
                print("Failed to create model instance.")
        else:
            print("Using existing model instance.")
        return cls._instance

    @classmethod
    def _create_instance(cls):
        # Initialize your model here. Replace 'YourModelClass' with your actual model class.
        #-----------
        CFG = {
            'input_size': const.INPUT_SIZE,
            'backbone_name': 'resnet18',
            'flow_step': 8,
            'hidden_ratio': 1.0,
            'conv3x3_only': True,
            'batch_size' : 64
            }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
       
        def build_model(): 
            print("***************here --------------laoding model***********")
            model = fastflow.FastFlow(
                backbone_name=CFG["backbone_name"],
                flow_steps=CFG["flow_step"],
                input_size = CFG['input_size'],
                conv3x3_only=CFG["conv3x3_only"],
                hidden_ratio=CFG["hidden_ratio"],
            )
            logTraing ="Model A.D. Param22#: {}".format(
                    sum(p.numel() for p in model.parameters() if p.requires_grad)
                )
            print(logTraing)

            with open(const.OUTPUT_FILE_PATH, 'a') as file:
                    file.write(logTraing+'\n')
            return model
        #-----------
        weight = const.WEIGHT_PATH
        checkpoint = torch.load(weight)
        model = build_model()
        model.cuda()
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
