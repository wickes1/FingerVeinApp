from gooey import Gooey, GooeyParser
import numpy as np
import pickle
from record import record
from feature_extract import feature_extract
from evaluate_offline import evaluate_offline
from enroll import enroll
from identify import identify
from utils import make_folder


@Gooey(
    program_name="Finger Vein Tool",
    advanced=True,
    # navigation="SIDEBAR",
    navigation="TABBED",
    show_sidebar=True,
    default_size=(600, 600),
    required_cols=1,
    optional_cols=1,
)
def main():
    settings_msg = "Finger Vein App for Record/Enroll/Identify/Evaluation"
    parser = GooeyParser(description=settings_msg)
    subs = parser.add_subparsers(help="commands", dest="command")
    #################################################################################
    tab1 = subs.add_parser("Record")
    tab1Group = tab1.add_argument_group("Offline Video Recording", description="Video will save under ./record or ./record_second , complete OfflineFunctions before verify/identify ")

    tab1Group.add_argument(
        "--user",
        type=str,
        default="001",
    )
    tab1Group.add_argument(
        "--length",
        help="Unit in second, 10 frames/second, 0 -> 999 seconds",
        type=int,
        default=15,
        widget="IntegerField",
    )
    fingerList = [
        "left_index",
        "left_middle",
        "left_ring",
        "right_index",
        "right_middle",
        "right_ring",
    ]
    tab1Group.add_argument("--finger", widget="Dropdown", default="left_index", choices=fingerList)
    tab1Group.add_argument("--second_session", action="store_true", default=False, help="record named with _2.avi")

    #################################################################################
    # tab2 = subs.add_parser("Enroll", help="direct")
    # tab2Group = tab2.add_argument_group("Realtime Enroll", description="Realtime enrollment will not save video and directly enroll user feature under ./pickle")
    # tab2Group.add_argument(
    #     "--user",
    #     type=str,
    #     default="001",
    # )
    # tab2Group.add_argument(
    #     "--length",
    #     help="Unit in second, 10 frames/second, 0 -> 999 seconds",
    #     type=int,
    #     default=15,
    #     widget="IntegerField",
    # )
    # tab2Group.add_argument("--finger", widget="Dropdown", default="left_index", choices=fingerList)
    #################################################################################
    tab3 = subs.add_parser("Identify")
    tab3Group = tab3.add_argument_group("Realtime Identify", description="At least enroll 1 user first, either realtime enroll or record video and use OfflineFunctions")

    try:
        with open("./pickle/enrolled_user.pkl", "rb") as f:
            features = pickle.load(f)
        if len(features) != 0:
            features = list(next(iter(features.values())).keys())
    except FileNotFoundError:
        features = []
        print("enrolled_user.pkl not found, pls enroll user first")
    # tab3Group2 = tab3.add_argument_group("Enrolled User List", description="\n".join([str(elem) for elem in features]))
    # tab3Group2.add_argument("--redundant")
    # tab3Group.add_argument(
    #     "--verify",
    #     widget="FilterableDropdown",
    #     choices=features,
    #     nargs="*",
    #     help="Choose the enrolled user for verify",
    # )
    # tab3Group.add_argument("--identify", action="store_true", default=False, help="Identify with all enrolled user")
    tab3Group.add_argument("--threshold", widget="DecimalField", default=0.8, help="Identify user threshold")

    #################################################################################
    tab4 = subs.add_parser("OfflineFunctions")
    tab4Group = tab4.add_argument_group(
        "Offline Feature Extraction, Enrollment and Evaluation",
        description="To extract feature or evaluate from recorded video, make sure you have record both session's video and extract feature before evaluation",
    )
    tab4Group.add_argument("--FeatureExtract", action="store_true", default=False, help="extract feature from record video ,save at ./pickle")
    tab4Group.add_argument("--FeatureExtract2", action="store_true", default=False, help="extract feature from record video 2nd session, save at ./pickle")
    tab4Group.add_argument("--Enrollment", action="store_true", default=False, help="Enroll 1st session feature, save at ./pickle")
    tab4Group.add_argument("--Evaluation", action="store_true", default=False, help="Evaluate all enrolled user with record video 2nd session, save at ./evaluation_result")
    tab4Group.add_argument("--ShowPlot", action="store_true", default=True, help="Show ROC and Hist in Evaluation")
    #################################################################################
    tab5 = subs.add_parser("PickleInfo")
    tab5Group = tab5.add_argument_group("Pickle information", description="Read info from enrolled_user/record_feature/record_feature_second")
    tab5Group.add_argument("--Record_Feature", action="store_true", default=True, help="./pickle/record_feature.pkl")
    tab5Group.add_argument("--Record_Feature_second", action="store_true", default=True, help="./pickle/record_feature_second.pkl")
    tab5Group.add_argument("--Enrolled_User_Method", action="store_true", default=True, help="./pickle/enrolled_user.pkl")
    #################################################################################
    args = parser.parse_args()
    print(args)
    print("*" * 80)
    if args.command == "Record":
        if args.second_session:
            record(args.user, args.length, args.finger + "_2", path="./record_second/")
        else:
            record(args.user, args.length, args.finger)
    # elif args.command == "Enroll":
    #     record(args.user, args.length, args.finger)
    #     feature_extract()
    elif args.command == "Identify":
        identify(threshold=float(args.threshold))
    elif args.command == "OfflineFunctions":
        if args.FeatureExtract:
            feature_extract()
        if args.FeatureExtract2:
            feature_extract(vidPath="./record_second/", pklPath="./pickle/record_feature_second.pkl")
        if args.Enrollment:
            enroll()
        if args.Evaluation:
            evaluate_offline(showPlot=args.ShowPlot)
    elif args.command == "PickleInfo":
        from pickle_info import pickle_info

        pickle_info(args.Enrolled_User_Method, args.Record_Feature, args.Record_Feature_second)


if __name__ == "__main__":
    make_folder()
    main()