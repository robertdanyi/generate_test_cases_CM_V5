# -*- coding: utf-8 -*-
"""
Task:
    Generate 'test case arrangements' (list of dictionaries) for experiment with a word and 3 target/distraction objects
    from the familiarisation video configurations (which are encoded in the video titles).

32 Familiarisation videos (2 objects and 1 word in each):
    - 16 'same' (same object on both sides)
        - 8 pointing (4 to the left, 4 to the right)
        - 8 nonpointing
    - 16 'diff' (2 different objects)
        - 8 pointing (4 to the left, 4 to the right)
        - 8 nonpointing

28 test cases (organised in 13 test labels):
-- 8 'same', where target object is picked from the fam. video with the same word:
    same_pointed_left (2), same_pointed_right (2), same_nonpoint_left (2), same_nonpoint_right (2)
-- 4 'same' with no target object: all 3 objects are from other videos than the word:
    same_nonpoint_notarget (2), same_pointing_notarget (2)
--12 'diff', where target object is picked from the fam. video with the same word:
    diff_pointed_left (2), diff_pointed_right (2), diff_nonpoint_left (2), diff_nonpoint_right (2),
    diff_unpointed_left (2), diff_unpointed_right (2)
-- 4 'diff' with no target object: all 3 objects are from other videos than the word:
    diff_nonpoint_notarget (4)

test label explanations:
    x_pointed_side => target obj in test arrangement is the object which is originally pointed at and is on 'side'
    x_unpointed_side => target is the object which is originally NOT pointed at in a pointing video, and is on 'side'
    x_nonpoint_side => target is from a non pointing video on 'side' (any side if x=same)
    x_y_notarget => all 3 objects are from other videos than the word in the test case

Constraints:
    - each word can be used in only one test case
    - each object can be reused max once (in another test case)
        (48 objects alltogether in group, will need 84 object occurences for test cases)

NOTE:
    - in the video names, 'right' label means the pointed object is on the LEFT (from the viewer's point of view)
    -> change perspective in 'extract_data' function

"""
import os
import pickle
import numpy as np
import pprint
from collections import Counter


DIR = os.path.dirname(os.path.abspath(__file__))
group_folders = [ entry.path for entry in os.scandir(DIR) if entry.name.startswith("group") and entry.is_dir() ]
save = False


def generate_arrangements(from_pickles=True):

    if from_pickles:
        group_names = ["group1A"] #, "group1B", "group2A", "group2B"]
        for group_name in group_names:
            arrangements = load_pickle(group_name)

            print(f"\nTesting {group_name} arrangements:")
            check_the_results(arrangements, group_name)

            print(f"\nArrangements for {group_name}:\n")
            for arr in arrangements:
                print("\n")
                pprint.pprint(arr)
            # pprint.pprint(arrangements)

    else:
        for group_folder in group_folders:
            group_name = os.path.basename(group_folder)
            print(f"\n\nExtracting data for {group_name}...")
            arrangements =  extract_data(group_folder)


            print(f"\nTesting {group_name} arrangements:")
            error = check_the_results(arrangements, group_name)

            # print(f"\nArrangements for {group_name}:\n")
            # pprint.pprint(arrangements)
            if ((not error)and save):
                save_to_pickle(arrangements, group_name)


def load_pickle(group_name):

    pickle_file = os.path.join(DIR, f"{group_name}_arrangements.pickle")
    arrangements = None
    with open (pickle_file, "rb") as handle:
        arrangements = pickle.load(handle)
        print(f"{group_name}_arrangements.pickle is loaded." )

    return arrangements


def extract_data(group_folder):
    """ NOTE: changing perspective from pointer agent (Bojana) as coded in video names,
        to viewer's point of view: rigth -> LEFT
    """
    all_video_names_in_group = [ entry.name.lower() for entry in os.scandir(group_folder) if entry.is_file() and
                              entry.name.endswith("mp4") ]

    assert len(all_video_names_in_group) == 32, "Something's wrong, there should be 32 video names"
    # pprint.pprint(all_vid_names_in_group)

    video_dicts = list(map(create_video_dict, all_video_names_in_group))

    assert len([d for d in video_dicts if d["cat"]=="same"]) == 16, "Number of 'same' dicts should be 16!"
    assert len([d for d in video_dicts if d["cat"]=="diff"]) == 16, "Number of 'diff' dicts should be 16!"

    arrangements = select_and_label_video_dictionaries(video_dicts, all_video_names_in_group) # dicts with list of dicts as values
    return arrangements


def create_video_dict(video_name: str) -> dict:
    """ Creates a dictionary from the values in the video title """

    keys = ["name", "word", "cat", "objs",
                "pointing", "pointed_LEFT", "pointed_RIGHT", "pointed_obj", "not_pointed_obj"]

    d = dict.fromkeys(keys)
    group, category_look, side, refs, word_ext = video_name.lower().split("_")
    category, look = category_look.split(".")
    word = word_ext.split(".")[0]

    d["name"] = video_name
    d["word"] = word
    d["cat"] = category
    d["objs"] = refs.split("-")
    d["pointing"] = side != "nop"
    if d["pointing"]:
        # CHANGING PERSPECTIVE: originally "right" pointing label will be "pointed_left"
        d["pointed_LEFT"] = side=="right"
        d["pointed_RIGHT"] = side=="left"
        if d["cat"]=="same":
            d["pointed_obj"] = d["objs"][0]
        else:
            d["pointed_obj"] = d["objs"][0] if d["pointed_LEFT"] else d["objs"][1]
            d["not_pointed_obj"] = d["objs"][1] if d["pointed_LEFT"] else d["objs"][0]

    return d


def select_and_label_video_dictionaries(video_dicts: list , all_video_names_in_group: list) -> list:
    """
    Random select n cases from the video dictionaries for each required test label:
    There will be 28 selections altogether, with 13 test labels.
    test_labels = video_labels + ["same_nonpoint_notarget", "same_pointing_notarget",
                                  "diff_nonpoint_notarget",
                                  "diff_unpointed_right", "diff_unpointed_left"]
    """
    video_labels = ["same_pointed_left", "same_pointed_right", "same_nonpoint_left", "same_nonpoint_right",
        "diff_pointed_left", "diff_pointed_right", "diff_nonpoint_left", "diff_nonpoint_right"]

    selections_by_video_label = {}
    for label in video_labels:
        n = 4 if (label.startswith("diff"))  else 3
        selections_by_video_label[label], video_dicts = select_videos(label, video_dicts, n)

    selections_by_test_label = {}
    for label in video_labels:
        selections_by_test_label[label] = selections_by_video_label[label][:2]

    # test labels that are not in video_labels
    selections_by_test_label["same_nonpoint_notarget"] = (selections_by_video_label["same_nonpoint_left"][2:3] +
                                                             selections_by_video_label["same_nonpoint_right"][2:3])
    selections_by_test_label["same_pointing_notarget"] = (selections_by_video_label["same_pointed_left"][2:3] +
                                                             selections_by_video_label["same_pointed_right"][2:3])
    selections_by_test_label["diff_nonpoint_notarget"] = (selections_by_video_label["diff_nonpoint_left"][2:4] +
                                                             selections_by_video_label["diff_nonpoint_right"][2:4])
    selections_by_test_label["diff_unpointed_right"] = selections_by_video_label["diff_pointed_left"][2:4]
    selections_by_test_label["diff_unpointed_left"] = selections_by_video_label["diff_pointed_right"][2:4]

    assert sum(map(len, selections_by_test_label.values())) == 28, "Nr of combinations should be 28!"
    assert len(selections_by_test_label.keys()) == 13, "Nr of labels should be 13!"

    arrangements = create_final_test_arrangements(selections_by_test_label, all_video_names_in_group)
    return arrangements


def select_videos(label: str, video_dicts: list, n: int) -> list:
    """ Selects n videos (dicts) according to the given label """

    category, pointing, side = label.split("_")

    if pointing == "nonpoint":
        # for 'same', side doesn't play a role in chosing the video, it is only labelled for placement
        lst_of_selected_vids = [ vid for vid in video_dicts if (vid["cat"]==category and vid["pointing"]==False)]
        lst_of_selected_vids = np.random.choice(lst_of_selected_vids, n, replace=False).tolist()
        video_dicts = [ vid for vid in video_dicts if vid not in lst_of_selected_vids ]

    else:
        point_cond = "pointed_{0}".format(side.upper())

        lst_of_selected_vids = [ vid for vid in video_dicts if (vid["cat"]==category and vid["pointing"]) ]
        lst_of_selected_vids = [ vid for vid in lst_of_selected_vids if vid[point_cond] ]
        lst_of_selected_vids = np.random.choice(lst_of_selected_vids, n, replace=False).tolist()
        video_dicts = [ vid for vid in video_dicts if vid not in lst_of_selected_vids ]

    # print(f"Nr of '{label}': {len(lst_of_selected_vids)} -- Remaining video dicts: {len(video_dicts)}")
    return lst_of_selected_vids, video_dicts


def create_final_test_arrangements(selections_by_test_label: dict, all_video_names_in_group: list) -> list:
    """
    arguments:
        selections_by_test_label: test label : selected list of video dictionaries
        all_video_names_in_group: the familiarisation video titles in the group
    returns:
        list of dicts

    For each test case (28), creates a dict (= arrangement):
        {label: label, word: word, target: ref,
             target_place: place,
             other1: ref, other1_place: place, other2: ref, other2_place: place]} or
        {label: label, word: word,
             other1: ref, other1_place: place,
             other2: ref, other2_place: place, other3: ref, other3_place: place]}

         where
            - label is one of the test labels
            - ref is object reference number
            - place is 0 (left), 1 (right) or 2 (middle)
    """

    all_obj_refs = get_all_objects(all_video_names_in_group)
    available_refs = all_obj_refs
    targets = [] # collect target objects to exclude them from available refs
    arrangements = []
    for label, selection in selections_by_test_label.items(): # each selection is a list of dicts

        side = label.split("_")[-1]
        if side != "notarget":
            for video_dict in selection: # 2 dicts in each selection; except 'unpoint'
                arr = dict(label=label, word=video_dict["word"].upper(),
                           original_objs=video_dict["objs"], # propagate the originally used objects to exclude from 'others'
                           target=get_target_obj_value(label, video_dict) # add target
                           )
                targets.append(arr["target"])
                arrangements.append(arr)

        # NOTARGET
        else:
            for video_dict in selection:
                arr = dict(label=label, word=video_dict["word"].upper(),
                           original_objs=video_dict["objs"]) # propagate the originally used objects to exclude from 'others'
                arrangements.append(arr)

    # add placement values
    arrangements = list(map(get_place_values, arrangements))
    # add other objects
    available_refs = [ ref for ref in available_refs if ref not in targets ]
    final_arrangements = populate_with_other_objects(arrangements, available_refs, all_obj_refs)

    assert len(arrangements) == 28, f"There should be 28 arrangements, not {len(arrangements)}!"

    return final_arrangements


def populate_with_other_objects(arrangements, available_refs, all_obj_refs):
    """
    Randomly chooses distractor objects (= others) for each arrangement
    from the pool of available objects.
    Excluded from the pool are:
    - the objects originally used with the word
    - all target objects used in test cases (only until refill)
    - all distractor objects already used in test cases
    """
    # print("\nPopulating arrangements with 'other' objects.")
    for arr in arrangements:
        n = 3 if arr["label"].endswith("notarget") else 2

        # print("\tSize of available pool:", len(available_refs))
        # available_refs will run out at some point, need refill
        pool = [ ref for ref in available_refs if ref not in arr["original_objs"] ]
        if len(pool) < n:
            available_refs = all_obj_refs
            # print("\t available_refs pool refilled\n")
            pool = [ ref for ref in available_refs if ref not in arr["original_objs"] ]

        others = np.random.choice(pool, n, replace=False).tolist()

        for i in range(n):
            arr[f"other{i+1}"] = others[i]

        # exclude the chosen 'other' objects from the pool to make sure they are not picked again
        available_refs = [ ref for ref in available_refs if ref not in others ]

    return arrangements


def get_all_objects(vid_names):
    """ Collects the references of all the objects used in the group """
    all_objects = []
    for vid in vid_names: # e.g. '1A_Diff.RL_Right_318-324_LEMI.mp4'
        objs = vid.split("_")[-2].split("-") # list
        all_objects += objs

    return all_objects


def get_target_obj_value(label: str, video_dict: dict) -> dict:
    """ Determines the target obj reference for an arrangement """
    cat, point, side = label.split("_")

    if point=="pointed":
        target = video_dict["pointed_obj"]
    elif point=="unpointed": # only diff
        target = video_dict["not_pointed_obj"]
    elif point=="nonpoint":
        if cat=="same":
            target = video_dict["objs"][0] # side don't matter
        else:
            target = video_dict["objs"][0] if side=="left" else video_dict["objs"][1]

    return target


def get_place_values(arr):
    """ Determines the object placement values (0: left, 1: middle, 2: right) """

    side = arr["label"].split("_")[-1]
    places = [0,1,2]
    arr["other1_place"] = places.pop(np.random.choice([0,1,2]))
    arr["other2_place"] = places.pop(np.random.choice([0,1]))
    if side == "notarget":
        arr["other3_place"] = places[0]
    else:
        arr["target_place"] = places[0]

    return arr


def save_to_pickle(arrangements, group_name):

    arrangements_pickle = os.path.join(DIR, f"{group_name}_arrangements.pickle")
    with open (arrangements_pickle, "wb") as handle:
        pickle.dump(arrangements, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{group_name}_arrangements.pickle is saved" )


def check_the_results(arrangements, group_name):
    """
    Checki if the list of dictionaries in the pickle is valid:
        - no doubles within a dict
        - each obj ref is repeated max once (2 occurences)
        - each word occurs only once

    """
    # 1. no doubles within a dict
    error = False
    all_refs = Counter()
    for d in arrangements:
        # collect objects in dict
        objs = [d.get("target"), d.get("other1"), d.get("other2"), d.get("other3")]
        objs = [ o for o in objs if o is not None ]
        objs_counter = Counter(objs)
        for key, val in objs_counter.items():
            if val>1:
                print(f"--> object {key} occurs more than once with {d['word']}")
                error = True
        all_refs.update(objs_counter)

    # 2. each object occurs max 2 times
    print("\tNumber of arrangements in group: ", len(arrangements))
    print("\tNumber of objects used in group: ", len(all_refs.keys()))
    print("\tNumber of object occurences in group: ", sum(all_refs.values()))
    occured_more_than2 = { obj: occurence>2 for obj, occurence in all_refs.items() }
    oversampled_objs = [ obj for obj in occured_more_than2.keys() if occured_more_than2.get(obj)]
    if len(oversampled_objs) > 0:
        print(f"--> object(s): {oversampled_objs} occur(s) more than 2 times in this group!")
        error = True
        # get the relevant arrangements
        affected_words = [ d.get("word") for d in arrangements if (d.get("target") in oversampled_objs or
                                                                  d.get("other1") in oversampled_objs or
                                                                  d.get("other2") in oversampled_objs or
                                                                  d.get("other3") in oversampled_objs) ]
        affected_words = list(set(affected_words))
        print(f"\t-> The affected words are: {affected_words}")

    # 3. each word occurs only once
    nr_of_individual_words = len(set(map(lambda x:x["word"], arrangements)))
    if nr_of_individual_words != len(arrangements):
        error = True
        print(f"\t-> nr of individual words ({nr_of_individual_words}) is not equal to the nr of arrangements (28)")

    if error:
        print("!! There is an ERROR in this group!")
    else:
        print("--> NO ERRORS found in this group.")

    return error


if __name__ == "__main__":
    generate_arrangements(from_pickles=True)











































