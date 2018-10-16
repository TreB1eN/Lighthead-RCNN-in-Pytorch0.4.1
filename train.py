import argparse

from LightheadRCNN_Learner import LightHeadRCNN_Learner

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("-r", "--resume", help="whether resume from the latest saved model",action="store_true")
    parser.add_argument("-save", "--from_save_folder", help="whether resume from the save path",action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    learner = LightHeadRCNN_Learner(training=True)
    learner.fit(18, resume=args.resume, from_save_folder=args.from_save_folder)