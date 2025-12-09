

objs = ["cube", "cylinder"]

gripper_dic = ["two_finger", "new_gripper"]

models = ["logistic_regression", "svm", "forest", "all"]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    from Simulation.simulation import Simulation
    from Data.data import Data
    from Models.models import Logistic_Regression, SVM, Random_Forest, compare_models
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, default="cube", choices=objs, help="Type of object to use in simulation.")
    parser.add_argument("--gripper", type=str, default="two_finger", choices=gripper_dic, help="Type of gripper to use in simulation.")
    parser.add_argument("--visuals", type=str, default="no visuals", choices=["visuals", "no visuals"], help="Whether to show simulation visuals.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of simulation iterations to run.")
    parser.add_argument("--save_data", type=str2bool, default=True, help="Whether to save the simulation data to a CSV file.")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "train", "test"], help="Whether to run simulations or train model.")
    parser.add_argument("--model", type=str, default="logistic_regression", choices=models, help="Type of model to train.")
    parser.add_argument("--file_save", type=str, default=None, help="File name to save simulation data.")
    parser.add_argument("--train_points", type=int, default=120, help="Number of training data points.")
    parser.add_argument("--test_points", type=int, default=300, help="Number of testing data points.")
    parser.add_argument("--val_points", type=int, default=0, help="Number of validation data points.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in Random Forest.")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Whether to shuffle data before splitting.")
    parser.add_argument("--predict", type=int, default=0, help="Number of predictions to make after training.")


    args = parser.parse_args()
    # print(args.mode)

    if args.mode == "run":
        print("Running simulations...")
        sim = Simulation(args.iterations, object=args.object, gripper=args.gripper, visuals=args.visuals, file_save=args.file_save)
        sim.run_simulations(save=args.save_data)
    elif args.mode == "train":
        print("Training model...")
        data = Data()
        data.import_data("Data/" + (args.file_save if args.file_save is not None else f"{args.object}-{args.gripper}-data.csv"))
        if args.model == "logistic_regression":
            model = Logistic_Regression(data, train_points=args.train_points, test_points=args.test_points, val_points=args.val_points, shuffle=args.shuffle)
        elif args.model == "svm":
            model = SVM(data, train_points=args.train_points, test_points=args.test_points, val_points=args.val_points, shuffle=args.shuffle)
        elif args.model == "forest":
            model = Random_Forest(data, train_points=args.train_points, test_points=args.test_points, n_estimators=args.n_estimators, shuffle=args.shuffle, val_points=args.val_points)
        elif args.model == "all":
            models_list = [
                Logistic_Regression(data, train_points=args.train_points, test_points=args.test_points, val_points=args.val_points, shuffle=args.shuffle),
                SVM(data, train_points=args.train_points, test_points=args.test_points, val_points=args.val_points, shuffle=args.shuffle),
                Random_Forest(data, train_points=args.train_points, test_points=args.test_points, n_estimators=args.n_estimators, shuffle=args.shuffle, val_points=args.val_points)
            ]
            results = compare_models(models_list, data)
            
        
        if args.model == "all":
            for model_name, accuracy in results.items():
                print(f"{model_name} test accuracy: {accuracy:.2f}")
        else:
            model.fit()
            model.save_model(f"Models/saved_models/{args.object}_{args.gripper}_{args.model}_model.pkl")
            print(f"{args.model} test accuracy: {model.test():.2f}")
            disp = model.confusion()

            disp.plot()
            plt.title(f"Confusion Matrix for {args.object} with {args.gripper} gripper")
            plt.savefig(f"Models/saved_models/{args.object}_{args.gripper}_{args.model}_confusion_matrix.jpg")
            plt.show()

    elif args.mode == "test":
        print("Testing model...")
        # run simulation to generate new data
        # sim = Simulation(iterations=args.predict, object=args.object, gripper=args.gripper, visuals=args.visuals, file_save=args.file_save)
        # sim.run_simulations(save=True)


        data = Data()
        data.import_data("Data/" + (args.file_save if args.file_save is not None else f"{args.object}-{args.gripper}-data.csv"))
        
        if args.model == "logistic_regression":
            model = Logistic_Regression(data, train_points=0, test_points=0, val_points=0, shuffle=args.shuffle)
        elif args.model == "svm":
            model = SVM(data, train_points=0, test_points=0, val_points=0, shuffle=args.shuffle)
        elif args.model == "forest":
            model = Random_Forest(data, train_points=0, test_points=0, n_estimators=args.n_estimators, shuffle=args.shuffle, val_points=0)

        else:
            models_list = [
                Logistic_Regression(data, train_points=0, test_points=0, val_points=0, shuffle=args.shuffle),
                SVM(data, train_points=0, test_points=0, val_points=0, shuffle=args.shuffle),
                Random_Forest(data, train_points=0, test_points=0, n_estimators=args.n_estimators, shuffle=args.shuffle, val_points=0)
            ]
            # results = compare_models(models_list, data)
            # for model_name, accuracy in results.items():
            #     print(f"{model_name} test accuracy: {accuracy:.2f}")
            

        model.load_model(f"Models/saved_models/{args.object}_{args.gripper}_{args.model}_model.pkl")
        predictions = model.predict(data.data[["x", "y", "z", "roll", "pitch", "yaw"]], data.data["success"])
        print(f"Predictions on new data: {predictions}")

    # sim = Simulation(1000, object="cube", gripper="two_finger", visuals="no visuals")
    """object = cube, cylinder | gripper = new_gripper, two_finger | visuals = visuals, no visuals """
    # sim.run_simulations(save=False)
    # # sim.save_data(save=True)
    
    # data = sim.data

    
    # data = Data()
    # data.import_data("cylinder-new_gripper-data.csv")
    # # data.remove_nans()
    # data.visualise_data()
    # data.statistics()
    # logr = Logistic_Regression(data, train_points=120, test_points=300)
    # logr.fit()
    # print(logr.test())
    # print(logr.predict(logr.test_data[["x", "y", "z", "roll", "pitch", "yaw"]]))
    # print(logr.test_data["success"])

