from SimpleIntersectionDQNEnvironment import SimpleIntersectionDQNEnvironment 
from SimpleIntersectionRandomEnvironment import SimpleIntersectionRandomEnvironment 
from SimpleIntersectionPPOEnvironment import SimpleIntersectionPPOEnvironment 

if __name__ == "__main__":
    SimpleIntersectionDQNEnvironment().save_model()
    SimpleIntersectionPPOEnvironment().save_model()
    SimpleIntersectionRandomEnvironment().train_model()

    SimpleIntersectionDQNEnvironment().evaluate_model()
    SimpleIntersectionPPOEnvironment().evaluate_model()
    SimpleIntersectionRandomEnvironment().evaluate_model()