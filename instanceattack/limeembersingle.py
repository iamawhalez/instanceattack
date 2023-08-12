import os
import ember
import argparse
import lightgbm as lgb


def limeembersingle(filename):
    # prog = "classify_binaries"
    # descr = "Use a trained ember model to make predictions on PE files"
    # parser = argparse.ArgumentParser(prog=prog, description=descr)
    # parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    # parser.add_argument("-m", "--modelpath", type=str, default=None, required=True, help="Ember model")
    # parser.add_argument("binaries", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
    # args = parser.parse_args()
    modelname="/home/gan/Downloads/mailwaredata/ember/ember_2017_2/model.txt"

    if not os.path.exists(filename):
        parser.error("ember model {} does not exist".format(filename))
    # lgbm_model = lgb.Booster(model_file=args.modelpath)
    lgbm_model = lgb.Booster(model_file=modelname)
    # newimg= np.zeros((filename.shape[0],filename.shape[1],1))
    # putty_data = open(filename, "rb").read()

    # for binary_path in args.binaries:
    if not os.path.exists(filename):
        print("{} does not exist".format(filename))

    file_data = open(filename, "rb").read()
    score = ember.predict_sample(lgbm_model, file_data, 2)

        # if len(args.binaries) == 1:
    print(score)
    return score

        # else:
        #     print("\t".join((binary_path, str(score))))

limeembersingle('/home/gan/Desktop/12.exe')