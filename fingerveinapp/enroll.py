import os
import pickle
import enroll_method as em


def enroll():

    if not os.path.exists("./pickle/record_feature.pkl"):
        raise FileNotFoundError("record_feature.pkl not found, pls record first.")
    else:
        with open("./pickle/record_feature.pkl", "rb") as f:
            record_feature = pickle.load(f)
        if len(record_feature) == 0:
            raise ValueError("Empty video list, no record video or all video's features are extracted")

    if not os.path.exists("./pickle/enrolled_user.pkl"):
        enrolled_user = {}
    else:
        with open("./pickle/enrolled_user.pkl", "rb") as f:
            enrolled_user = pickle.load(f)

    for user, features in record_feature.items():
        print("enrolling {}".format(user))
        for method in em.singleTemplateMethodList:
            templateList = method(features)
            try:
                enrolled_user[method.__name__].update({user: templateList})
            except KeyError:
                enrolled_user.update({method.__name__: {user: templateList}})
        for method in em.mutliTemplateMethodList:
            for num_of_template in range(2, 6):
                templateList = method(features, num_of_template)
                try:
                    enrolled_user["{}_{}".format(num_of_template, method.__name__)].update({user: templateList})
                except KeyError:
                    enrolled_user.update({"{}_{}".format(num_of_template, method.__name__): {user: templateList}})

    with open("./pickle/enrolled_user.pkl", "wb") as f:
        pickle.dump(enrolled_user, f)


if __name__ == "__main__":

    enroll()