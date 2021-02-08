# -*- coding: utf-8 -*-
"""
Task:
    create arrangements (list of dictionaries) with the word and 3 target/distraction objects

WORDS:
28 test cases:
-- 8 'same': 3 objs, of which the 'target' obj is picked from the corresponding familiarization video:
    - 4x word and target obj is from pointing (2 point right, 2 point left)
    - 4x word and target obj is from non-pointing
-- 12 'diff': 3 objs, of which the 'target' obj is picked from the corresponding familiarization video:
    - 4x word and target is from pointing, obj is pointed (2 point right, 2 point left)
    - 4x word and target is from pointing but obj is NOT the pointed one (2 point right, 2 point left)
    - 4x word and target is from non-pointing, 2-2 from both sides of appearance)
-- 8 'notarget' all 3 objs are from other videos than the word
    - 2x word is from diff pointing
    - 2x word is from diff non-pointing,
    - 2x word is from same pointing,
    - 2x word is from same non-pointing

OBJECTS:
-> SIDE a tesztben a tárgyak 1:3 arányban legyenek ugyanazon az oldalon, ahol voltak a videókban (DONE as 2:2 in same)
-> BALANCING the 'other' objects- WE CANNOT REALLY

A videókban 16*2 + 16*1, azaz 48 tárgy szerepel, ebből 16-ra történik rámutatás
A tesztben összesen 84 tárgy lesz.
TODO: Azt kéne megoldani, hogy max 1x ismétlődjön egy tárgy.
TODO: the side problem:
    in the video name, right label means the pointed object is on the left (from the viewer's point of view)
    DONE: changing perspective in load_dataframe

"""
import os
# from collections import defaultdict, namedtuple
import pickle
import numpy as np
import pprint
from collections import Counter


DIR = os.path.dirname(os.path.abspath(__file__))
group_folders = [ entry.path for entry in os.scandir(DIR) if entry.name.startswith("group") and entry.is_dir() ]
all_video_names = {}
save = False


def generate(from_pickles=True):

    if from_pickles:
        group_names = ["group1A"] #, "group1B", "group2A", "group2B"]
        for group_name in group_names:
            arrangements = load_pickle(group_name)

            print(f"\nTesting {group_name} arrangements:")
            check_the_results(arrangements, group_name)

            print(f"\nArrangements for {group_name}:\n")
            pprint.pprint(arrangements)

    else:
        # group_names = ["group1A"] #, "group1B", "group2A", "group2B"]
        for group_folder in group_folders:
            group_name = os.path.basename(group_folder)
            print(f"\n\nExtracting data for {group_name}")
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

    group_name = os.path.basename(group_folder)
    all_video_names[group_name] = all_video_names_in_group

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
    Random select x from the video dictionaries for each required test label:
        - 8 same from 16 same
            - 4 pointing - done
            - 4 non pointing - done
        - 12 diff from 16 diff
            - 8 point (from 8) - done
            - 4 non-point (from 8 - 2 obj from left, 2 from right) - done
        - 4 same from 8 same (notarget)
            - 2 point
            - 2 nonpoint
        - 4 diff from 4 diff (notarget)
            - 4 nonpoint (no more pointing in group)

    There will be 28 selections altogether, with 13 labels.
    """
    labels = ["same_pointed_left", "same_pointed_right", "same_nonpoint_left", "same_nonpoint_right",
        "diff_pointed_left", "diff_pointed_right", "diff_nonpoint_left", "diff_nonpoint_right"]

    selections_by_label = {}
    for label in labels:
        n = 4 if (label.startswith("diff"))  else 3
        selections_by_label[label], video_dicts = select_videos(label, video_dicts, n)

    selections_by_test_category = {} # 14 keys, 26 values
    selections_by_test_category["same_nonpoint_left"] = selections_by_label["same_nonpoint_left"][:2]
    selections_by_test_category["same_nonpoint_right"] = selections_by_label["same_nonpoint_right"][:2]
    selections_by_test_category["same_nonpoint_notarget"] = (selections_by_label["same_nonpoint_left"][2:3] +
                                                 selections_by_label["same_nonpoint_right"][2:3])
    selections_by_test_category["same_pointed_left"] = selections_by_label["same_pointed_left"][:2]
    selections_by_test_category["same_pointed_right"] = selections_by_label["same_pointed_right"][:2]
    selections_by_test_category["same_pointing_notarget"] = (selections_by_label["same_pointed_left"][2:3] +
                                                 selections_by_label["same_pointed_right"][2:3])

    selections_by_test_category["diff_nonpoint_left"] = selections_by_label["diff_nonpoint_left"][:2]
    selections_by_test_category["diff_nonpoint_right"] = selections_by_label["diff_nonpoint_right"][:2]
    selections_by_test_category["diff_nonpoint_notarget"] = (selections_by_label["diff_nonpoint_left"][2:4] +
                                                 selections_by_label["diff_nonpoint_right"][2:4])

    selections_by_test_category["diff_pointed_left"] = selections_by_label["diff_pointed_left"][:2]
    selections_by_test_category["diff_pointed_right"] = selections_by_label["diff_pointed_right"][:2]
    selections_by_test_category["diff_unpointed_right"] = selections_by_label["diff_pointed_left"][2:4]
    selections_by_test_category["diff_unpointed_left"] = selections_by_label["diff_pointed_right"][2:4]
    # selections_by_test_category["diff_pointing_notarget"] = (selections_by_label["diff_pointed_left"][3:4] +
                                                 # selections_by_label["diff_pointed_right"][3:4])

    assert sum(map(len, selections_by_test_category.values())) == 28, "Nr of combinations should be 28!"
    assert len(selections_by_test_category.keys()) == 13, "Nr of labels should be 13!"

    arrangements = create_final_test_arrangements(selections_by_test_category, all_video_names_in_group)
    return arrangements


def select_videos(label: str, video_dicts: list, n: int) -> list:
    """
    select n videos (dicts)
    according to the given category (same/diff),
    pointing (pointing/nonpoint/unpoint) and
    side (left/right)
    """
    print(f"\nSelecting videos for label: {label}")
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

    print(f"Nr of '{label}': {len(lst_of_selected_vids)} -- Remaining video dicts: {len(video_dicts)}")
    return lst_of_selected_vids, video_dicts


def create_final_test_arrangements(selections_by_test_category: dict, all_video_names_in_group: list) -> list:
    """
    arguments:
        selections_by_test_category : a dictionary of form label:selected list of video dictionaries
        all_video_names_in_group: the video titles in the group

    returns:
        list of dicts

    For each test case, create a dict (= arrangement):
        {label: label, word: word, target: ref,
             target_place: place,
             other1: ref, other1_place: place, other2: ref, other2_place: place]} or
        {label: label, word: word,
             other1: ref, other1_place: place,
             other2: ref, other2_place: place, other3: ref, other3_place: place]}

         where
            - label is one of the keys of the selections dictionary
            - ref is object reference number
            - place is 0 (left), 1 (right) or 2 (middle)

    labels:
        same_pointed_left, same_pointed_right, same_nonpoint_left, same_nonpoint_right,
        same_nonpoint_notarget, same_pointing_notarget,
        diff_pointed_left, diff_pointed_right, diff_nonpoint_left, diff_nonpoint_right,
        diff_unpointed_left, diff_unpointed_right,
        diff_nonpoint_notarget, diff_pointing_notarget

    label meanings:
        x_pointed_side => target is pointed obj from 'side'
        x_unpointed_side => target is unpointed obj from 'side'
        x_nonpoint_side => target is obj from 'side' (any side if x=same)
        x_pointing_notarget => all 3 objects are from 'other_obj'
        x_nonpoint_notarget => all 3 objects are from 'other_obj'

    object availability:
        there are 48 objects in the group
        we need 8*3 + 10*3 + 8*3 = 78
        -> some will be repeated
    """
    all_obj_refs = get_all_objects(all_video_names_in_group)
    available_refs = all_obj_refs
    targets = [] # collect target objects to exclude them from available refs
    arrangements = []
    for label, selection in selections_by_test_category.items(): # each value is a list of dicts

        side = label.split("_")[-1]
        if side != "notarget":
            # fix = True
            for video_dict in selection: # 2 dicts in each selection; except 'unpoint'
                # init arrangement dictionary
                arr = dict(label=label, word=video_dict["word"].upper(), side=side, # fix=fix,
                           original_objs=video_dict["objs"], # propagate the originally used objects to exclude from 'others'
                           target=get_target_obj_value(label, video_dict) # add target
                           )
                targets.append(arr["target"])
                arrangements.append(arr)
                # switch to not fixed
                # fix=False

        # NOTARGET
        else:
            for video_dict in selection:
                # init arrangement dictionary
                arr = dict(label=label, word=video_dict["word"].upper(), side=side,
                           original_objs=video_dict["objs"]) # propagate the originally used objects to exclude from 'others'
                arrangements.append(arr)

    # add placement values
    arrangements = list(map(get_place_values, arrangements))
    # add other objects
    available_refs = [ ref for ref in available_refs if ref not in targets ]
    final_arrangements = populate_with_other_objects(arrangements, available_refs, all_obj_refs)

    assert len(arrangements) == 28, f"There should be 28 arrangements, not {len(arrangements)}!"

    # compare selections with arrangements wrt side constraints
    # test_side_constraints(selections_by_test_category, arrangements)

    return final_arrangements



def populate_with_other_objects(arrangements, available_refs, all_obj_refs):
    """ populate the not target objects (= others)
        from the pool of available objects, from which excluded are:
        - the objects used with the word
        - the other targets (until refill)
    """
    print("\nPopulating arrangements with 'other' objects.")
    for arr in arrangements:
        n = 3 if arr["label"].endswith("notarget") else 2

        print("\tSize of available pool:", len(available_refs))
        # available_refs will run out at some point, need refill
        if len(available_refs) < n:
            available_refs = all_obj_refs
            print("\t available_refs pool refilled\n")

        others = np.random.choice([ ref for ref in available_refs if ref not in arr["original_objs"] ],
                                  n, replace=False).tolist()
        for i in range(n):
            arr[f"other{i+1}"] = others[i]

        # exclude the chosen 'other' objects from the pool to make sure they are not picked again
        available_refs = [ ref for ref in available_refs if ref not in others ]

    return arrangements


def get_all_objects(vid_names):
    """ Collect references of all objects used in group """
    all_objects = []
    for vid in vid_names: # 1A_Diff.RL_Right_318-324_LEMI.mp4
        objs = vid.split("_")[-2].split("-") # list
        all_objects += objs

    return all_objects


def get_target_obj_value(label: str, video_dict: dict) -> dict:
    """ Determine target obj ref for arrangement """
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
    """ Determine object placement values """

    side = arr["side"]
    places = [0,1,2]
    arr["other1_place"] = places.pop(np.random.choice([0,1,2]))
    arr["other2_place"] = places.pop(np.random.choice([0,1]))
    if side == "notarget":
        arr["other3_place"] = places[0]
    else:
        arr["target_place"] = places[0]

    return arr


    # if side == "notarget":
    #     places = [0,1,2]
    #     arr["other1_place"] = places.pop(np.random.choice([0,1,2]))
    #     arr["other2_place"] = places.pop(np.random.choice([0,1]))
    #     arr["other3_place"] = places[0]

    #     return arr

    # if arr["fix"]:
    #     arr["target_place"] = 0 if side=="left" else 2
    #     places = [1,2] if side=="left" else [0,1]
    # else:
    #     places = [0,1,2]
    #     nofix_places = [1,2] if side=="left" else [0,1]
    #     arr["target_place"] = places.pop(np.random.choice(nofix_places)) # other than right
    # arr["other1_place"] = places.pop(np.random.choice([0,1]))
    # arr["other2_place"] = places[0]

    # return arr


def save_to_pickle(arrangements, group_name):

    arrangements_pickle = os.path.join(DIR, f"{group_name}_arrangements.pickle")
    with open (arrangements_pickle, "wb") as handle:
        pickle.dump(arrangements, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{group_name}_arrangements.pickle is saved" )


def check_the_results(list_of_dict, group_name):
    """
    Checki if the list of dictionaries on the pickle is valid:
        - no doubles within a dict
        - each obj ref is repeated max once (2 occurences)
        - each word occurs only once
        - sides are good

    """
    # 1. no doubles within a dict
    error = False
    all_refs = Counter()
    for d in list_of_dict:
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
    print("\tNumber of arrangements in group: ", len(list_of_dict))
    print("\tNumber of objects used in group: ", len(all_refs.keys()))
    print("\tNumber of object occurences in group: ", sum(all_refs.values()))
    for obj, occurence in all_refs.items():
        if occurence>2:
            print(f"--> object {obj} occurs more than 2 times in this group!")
            error = True
            # get the arrangement where it happens
            for d in list_of_dict:
                if d.get("target") == obj:
                    print(f"\t-> {obj} is the target in one of the {d['label']}")
                elif obj in [d.get("other1"), d.get("other2"), d.get("other3")]:
                    print(f"\t-> {obj} is one of the others in one of the {d['label']}")

    # 3. each word occurs only once
    nr_of_individual_words = len(set(map(lambda x:x["word"], list_of_dict)))
    if nr_of_individual_words != len(list_of_dict):
        error = True
        print(f"\t-> nr of individual words ({nr_of_individual_words}) is not equal to the nr of arrangements (28)")
    else:
        print(f"\tNumber of individual words in the arrangement: {nr_of_individual_words}")


    if error:
        print("!! There is an ERROR in this group!")
    else:
        print("--> NO ERRORS found in this group.")

    return error


def test_side_constraints(selections_by_test_category, arrangements):
    """
    to check:
        - target object side vs original object side (pointing and unpointing)
            - pointing: same side, opposite side
            - unpointing: same side, opposite side
        - there is no object from original video in nontarget arrangements
    """
    for label in list(selections_by_test_category.keys()):
        video_dicts_for_label = selections_by_test_category[label]
        for video_dict in video_dicts_for_label:
            word = video_dict["word"]
            arr = [ arr for arr in arrangements if arr["word"]==word ][0]

            side = video_dict["name"].lower().split("_")[2]
            side = 0 if side=="right" else 2 # taking into account the perspective change
            cat, point, side = arr["label"].split("_")
            if arr["label"].startswith("diff_pointed"):
                if arr["fix"]:
                    assert arr.get("target_place") == video_dict[""]




if __name__ == "__main__":
    generate(from_pickles=False)











































