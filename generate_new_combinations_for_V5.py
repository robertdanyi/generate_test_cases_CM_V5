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
save = False


def generate(from_pickles=True):

    if from_pickles:
        group_names = ["group1A"] #, "group1B", "group2A", "grou2B"]
        for group_name in group_names:
            arrangements = load_pickle(group_name)

            print(f"\nTesting {group_name} arrangements:")
            check_the_results(arrangements, group_name)

            print(f"\nArrangements for {group_name}:\n")
            pprint.pprint(arrangements)

    else:
        # group_names = ["group1A"] #, "group1B", "group2A", "grou2B"]
        for group_folder in group_folders:
            arrangements =  extract_data(group_folder)

            group_name = os.path.basename(group_folder)
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
    print("nr of arrangements>", len(arrangements))
    return arrangements


def create_video_dict(video_name):

    keys = ["name", "word", "cat", "objs",
                "pointing", "pointed_LEFT", "pointed_RIGHT", "pointed_obj", "not_pointed_obj"]

    d = dict.fromkeys(keys)
    group, cat_look, side, refs, word_ext = video_name.lower().split("_")
    cat, look = cat_look.split(".")
    word = word_ext.split(".")[0]

    d["name"] = video_name
    d["word"] = word
    d["cat"] = cat
    d["objs"] = refs.split("-")
    d["pointing"] = side != "nop"
    # d["side"] = side
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


def select_and_label_video_dictionaries(video_dicts, all_video_names_in_group):
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
            - 2 point (here it's interesting - already had 8 point above)
            - 2 nonpoint
    """
    selections_dict = {}
    # 4 same non-pointing
    # random choice of left, right, notarget
    same_nonpointing = [ vid for vid in video_dicts if (vid["cat"]=="same" and not vid["pointing"]) ]
    same_nonpointing = np.random.choice(same_nonpointing, 6, replace=False).tolist()
    selections_dict["same_nonpoint_left"] = same_nonpointing[:2]
    selections_dict["same_nonpoint_right"] = same_nonpointing[2:4]

    # 2 same non-pointing notarget
    selections_dict["same_nonpoint_notarget"] = same_nonpointing[4:6]

    # 4 same pointing
    same_pointing = [ vid for vid in video_dicts if (vid["cat"]=="same" and vid["pointing"]) ]
    same_pointing_left = np.random.choice([ vid for vid in same_pointing if vid["pointed_LEFT"] ], 3, replace=False).tolist()
    same_pointing_right = np.random.choice([ vid for vid in same_pointing if vid["pointed_RIGHT"] ], 3, replace=False).tolist()
    selections_dict["same_pointed_left"] = same_pointing_left[:2]
    selections_dict["same_pointed_right"] = same_pointing_right[:2]

    # 2 same pointing notarget
    selections_dict["same_pointing_notarget"] = [same_pointing_left[2], same_pointing_right[2]]

    # 8 diff pointing
    diff_pointing = [ vid for vid in video_dicts if (vid["cat"]=="diff" and vid["pointing"]) ]
    diff_pointing_left = np.random.choice([ vid for vid in diff_pointing if vid["pointed_LEFT"] ], 4, replace=False).tolist()
    diff_pointing_right = np.random.choice([ vid for vid in diff_pointing if vid["pointed_RIGHT"] ], 4, replace=False).tolist()
    selections_dict["diff_pointed_left"] = diff_pointing_left[:2]
    selections_dict["diff_unpointed_right"] = diff_pointing_left[2:4]
    selections_dict["diff_pointed_right"] = diff_pointing_right[:2]
    selections_dict["diff_unpointed_left"] = diff_pointing_right[2:4]

    # 2 diff pointing notarget -> must sample 2 again, as all 8 are used
    diff_pointing_notarget = [np.random.choice(diff_pointing_left),
                              np.random.choice(diff_pointing_right)]
    selections_dict["diff_pointing_notarget"] = diff_pointing_notarget

    # 4 diff nonpointing
    diff_nonpointing = [ vid for vid in video_dicts if (vid["cat"]=="diff" and not vid["pointing"]) ]
    diff_nonpointing = np.random.choice(diff_nonpointing, 6, replace=False).tolist()
    selections_dict["diff_nonpoint_left"] = diff_nonpointing[:2]
    selections_dict["diff_nonpoint_right"] = diff_nonpointing[2:4]

    # 2 diff nonpoint notarget
    selections_dict["diff_nonpoint_notarget"] = diff_nonpointing[4:6]

    assert sum(map(len, selections_dict.values())) == 28, "Nr of combinations should be 28!"
    assert len(selections_dict.keys()) == 14, "Nr of labels should be 14!"

    arrangements = create_final_test_arrangements(selections_dict, all_video_names_in_group)
    return arrangements


def create_final_test_arrangements(selections_dict, all_video_names_in_group):
    """
    arguments:
        selections_dict : a dictionary of form label:selected list of video dictionaries
        all_video_names_in_group: the video titles in the group

    For each case in each group, create a dict:
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
        we need 8*3 + 12*3 + 8*3 = 84
        -> some will be repeated
    """
    all_obj_refs = get_all_objects(all_video_names_in_group)
    available_refs = all_obj_refs
    arrangements = []
    for label, selection in selections_dict.items(): # each value is a list of dicts

        cat, point, side = label.split("_")
        if side != "notarget":
            fix = True
            for video_dict in selection: # 2 dicts in each selection
                arr, available_refs = arrange_to_dict(label, video_dict, fix, available_refs)
                fix=False
                arrangements.append(arr)

        # NOTARGET
        else:
            for video_dict in selection:
               arr = {"label": label, "word": video_dict["word"].upper()}
               places = [0,1,2]
               arr["other1_place"] = places.pop(np.random.choice([0,1,2]))
               arr["other2_place"] = places.pop(np.random.choice([0,1]))
               arr["other3_place"] = places[0]
               arrangements.append(arr)

    assert len(arrangements) == 28, f"There should be 28 arrangements, not {len(arrangements)}!"

    arrangements = populate_with_other_objects(arrangements, available_refs, all_obj_refs)
    # arrangements = [ populate_with_other_objects(arr, available_refs, all_obj_refs) for arr in arrangements ]
    ###### can this be done in a functional way? List compr or map? The extra args...

    return arrangements



def populate_with_other_objects(arrangements, available_refs, all_obj_refs):
    """ populate the not target objects (= others) """

    for arr in arrangements:
        # available_refs will run out at some point, need refill
        if len(available_refs) < 2:
            available_refs = all_obj_refs
            print("\n**refill**\n")

        n = 3 if arr["label"].endswith("notarget") else 2
        others = np.random.choice([ ref for ref in available_refs if ref != arr.get("target") ],
                                  n, replace=False).tolist()
        arr["other1"] = others[0]
        arr["other2"] = others[1]
        if n == 3:
            arr["other3"] = others[2]
        available_refs = [ ref for ref in available_refs if ref not in others ]

    return arrangements


def get_all_objects(vid_names):

    all_objects = []
    for vid in vid_names: # 1A_Diff.RL_Right_318-324_LEMI.mp4
        objs = vid.split("_")[-2].split("-") # list
        all_objects += objs

    return all_objects


def arrange_to_dict(label, video_dict, fix, available_refs):
    """ Should create the target and the places,
        so that the other objects can be picked from the available pool afterwards
        TO TEST:
            what if the label doesn't match the video_dict?"""

    cat, point, side = label.split("_")

    arr = {"label": label, "word": video_dict["word"].upper(), "fix":fix}

    if point=="pointed":
        arr["target"] = video_dict["pointed_obj"]
    elif point=="unpointed": # only diff
        arr["target"] = video_dict["not_pointed_obj"]
    elif point=="nonpoint":
        if cat=="same":
            arr["target"] = video_dict["objs"][0]
        else:
            arr["target"] = video_dict["objs"][0] if side=="left" else video_dict["objs"][1]

    # remove the target obj from the pool of available objects
    available_refs = [ ref for ref in available_refs if ref != arr["target"] ]

    if fix:
        arr["target_place"] = 0 if side=="left" else 2
        places = [1,2] if side=="left" else [0,1]
        fix = False
    else:
        places = [0,1,2]
        nofix_places = [1,2] if side=="left" else [0,1]
        arr["target_place"] = places.pop(np.random.choice(nofix_places)) # other than right
    arr["other1_place"] = places.pop(np.random.choice([0,1]))
    arr["other2_place"] = places[0]

    return arr, available_refs


def find_others(arr, available_refs):

    n = 3 if arr["label"].endswith("notarget") else 2
    others = np.random.choice( [ ref for ref in available_refs if ref != arr.get("target") ], n, replace=False ).tolist()
    arr["other1"] = others[0]
    arr["other2"] = others[1]
    if n == 3:
        arr["other3"] = others[2]
    available_refs = [ ref for ref in available_refs if ref not in others ]

    return arr, available_refs


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
    nr_of_individual_words = len(list(map(lambda x:x["word"], list_of_dict)))
    if nr_of_individual_words != len(list_of_dict):
        error = True
        print(f"\t-> nr of individual words ({nr_of_individual_words}) is not equal to the nr of arrangements (28)")
    else:
        print(f"\tNumber of individual words in the arrangement: {nr_of_individual_words}")

    # 4. sides are as exected


    if error:
        print("!! There is an ERROR in this group!")
    else:
        print("--> NO ERRORS found in this group.")

    return error


if __name__ == "__main__":
    generate(from_pickles=True)











































