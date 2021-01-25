import pickle


def pickle_info(Enrolled_User=True, Record_Feature=True, Record_Feature_second=True):

    if Record_Feature:
        try:
            with open("./pickle/record_feature.pkl", "rb") as f:
                features = pickle.load(f)
            print("./pickle/record_feature.pkl")
            print("-" * 50)
            for key, item in features.items():
                print(key, type(item), item.shape)
        except:
            print("./pickle/record_feature.pkl Not Found")
        print("*" * 80)

    if Record_Feature_second:
        try:
            with open("./pickle/record_feature_second.pkl", "rb") as f:
                features = pickle.load(f)
            print("./pickle/record_feature_second.pkl")
            print("-" * 50)
            for key, item in features.items():
                print(key, type(item), item.shape)
        except:
            print("./pickle/record_feature_second.pkl Not Found")
        print("*" * 80)

    if Enrolled_User:
        try:
            with open("./pickle/enrolled_user.pkl", "rb") as f:
                features = pickle.load(f)
            print("./pickle/enrolled_user.pkl")
            print("-" * 50)
            print("Enrolled Methods:")
            print("\n".join(features.keys()))
            print("=" * 50)
            print("Enrolled User:")
            print("\n".join(features[list(features.keys())[0]].keys()))
            # for key, item in features.items():
            #     print("=" * 50)
            #     print(key, type(item))
            #     for key2, item2 in item.items():
            #         print(key2, type(item2))
        except:
            print("./pickle/enrolled_user.pkl Not Found")
        print("*" * 80)


if __name__ == "__main__":
    pickle_info()