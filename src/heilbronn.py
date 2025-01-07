from HeilbronnDQNEnvironment import HeilbronnDQNEnvironment 
from HeilbronnRandomEnvironment import HeilbronnRandomEnvironment 
from HeilbronnPPOEnvironment import HeilbronnPPOEnvironment 

if __name__ == "__main__":
    HeilbronnDQNEnvironment().save_model()
    HeilbronnPPOEnvironment().save_model()
    HeilbronnRandomEnvironment().train_model()

    HeilbronnDQNEnvironment().evaluate_model()
    HeilbronnPPOEnvironment().evaluate_model()
    HeilbronnRandomEnvironment().evaluate_model()