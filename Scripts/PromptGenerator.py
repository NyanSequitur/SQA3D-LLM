import json
import re
from tqdm.auto import tqdm
import spacy
from nltk.corpus import wordnet as wn
import json
from textblob import TextBlob
from nltk import pos_tag
import tqdm
from inflect import engine
import itertools
import openai
import os
from time import sleep

inflect = engine()


from dotenv import load_dotenv

# load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

import logging

# log to log.log
logging.basicConfig(
    filename="log.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


# ===================================================================================================
# LEGACY FUNCTIONS
# ===================================================================================================


def isObject(word):
    # exceptions for words that *are* objects, but aren't useful for our purposes
    # the goal is to identify useful objects, and walls and floors are not useful.
    alwaysFalseLocations = ["floor", "wall", "ceiling"]
    misclassifiedNouns = ["nap", "writing", "coffee", "does", "shape"]
    missedNouns = ["whiteboard", "door"]
    if word in missedNouns:
        return True

    if word == "object":
        return False
    alwaysFalse = alwaysFalseLocations + misclassifiedNouns
    object_synset = wn.synsets("object")[0]
    entity_synset = wn.synsets("entity")[0]
    # print("Object Synset:", object_synset)
    for ss in wn.synsets(word):
        # check if its a location
        if isLocationSS(ss):
            return False
        # Get the lowest common hypernym
        hit = object_synset.lowest_common_hypernyms(ss, use_min_depth=True)
        # print(hit)
        if hit and (hit[0] == object_synset) and word not in alwaysFalse:
            return True
    return False


def isLocationNounPhrase(nounPhrase):
    words = nounPhrase.split()
    object_synset = wn.synsets("object")[0]
    location_synset = wn.synsets("location")[0]
    # for each word in the noun phrase, if it is a location, return True
    for word in words:
        for ss in wn.synsets(word):
            if isLocationSS(ss):
                return True
    return False


def isLocationSS(
    ss, object_synset=wn.synsets("object")[0], location_synset=wn.synsets("location")[0]
):
    # Get the lowest common hypernym
    hit = location_synset.lowest_common_hypernyms(ss, use_min_depth=True)
    if not hit:
        return False
    # get depth of lowest common hypernym
    depth = location_synset.shortest_path_distance(ss)
    # return False if location is shallower than object
    if depth < 0:
        return False
    if hit and hit[0] == location_synset:
        return True
    return False


def getNouns(question):
    # Remove Q, and Context: from the question
    question = question.replace("Q: ", "")
    question = question.replace("Context: ", "")

    blob = TextBlob(question)
    nouns = []
    nounPhrases = []
    # identify nouns and noun phrases
    # being careful not to count nouns that are part of noun phrases twice
    nounPhrases = blob.noun_phrases

    # strip adjectives, adverbs, and determiners from noun phrases
    for i, nounPhrase in enumerate(nounPhrases):
        nounPhrases[i] = stripAdjAdvDet(nounPhrase)
        # if one word remains, move it from nounPhrases to nouns
        if len(nounPhrases[i].split()) == 1:
            nouns.append(nounPhrases[i])
            nounPhrases.remove(nounPhrases[i])

    for word, tag in pos_tag(blob.words):
        if tag == "NN" and word not in "".join(nounPhrases):
            nouns.append(word)

    # get nouns using spacy
    doc = nlp(question)
    for token in doc:
        if (
            token.pos_ == "NOUN"
            and token.text not in "".join(nouns)
            and token.text not in "".join(nounPhrases)
        ):
            nouns.append(token.text)

    return nounPhrases, nouns


def stripAdjAdvDet(nounPhrase):
    blob = TextBlob(nounPhrase)
    for word, tag in pos_tag(blob.words):
        if tag == "JJ" or tag == "RB" or tag == "DT":
            nounPhrase = nounPhrase.replace(word, "")
    return nounPhrase.strip()


def getNounPairs(question, nlp=spacy.load("en_core_web_sm")):
    doc = nlp(question)
    nounPhrases = []
    for i in range(len(doc) - 1):
        if doc[i].pos_ == "NOUN" and doc[i + 1].pos_ == "NOUN":
            nounPhrases.append(f"{doc[i].text} {doc[i + 1].text}")

    return nounPhrases


def readCaptions(filename, limit):
    with open(filename) as f:
        # read the captions
        captions = f.readlines()
        # split captions at periods
        # captions = [caption.split(".") for caption in captions]
        # flatten the list
        # captions = list(itertools.chain.from_iterable(captions))
        # strip newlines, remove empty strings
        captions = [caption.strip() for caption in captions if caption.strip()]

        # capitalize the first letter of each caption and add a period to the end
        for i, caption in enumerate(captions):
            captions[i] = caption.capitalize()
            if captions[i][-1] != ".":
                captions[i] += "."

        # split captions into a number of chunks equal to the limit, with the last chunk containing the remaining captions. Then, interleave the chunks.
        for i in range(len(captions)):
            captions[i] = captions[i].split()
        captions = [captions[i : i + limit] for i in range(0, len(captions), limit)]
        captions = list(
            itertools.chain.from_iterable(
                itertools.zip_longest(*captions, fillvalue="")
            )
        )
        # remove empty strings
        captions = [caption for caption in captions if caption]
        # join the captions back together
        captions = [" ".join(caption) for caption in captions]
    return captions


# ===================================================================================================
# END LEGACY FUNCTIONS
# ===================================================================================================


def limit_nouns(question, limit=4):
    # selects the first limit number of captions that contain each noun
    # if there are fewer than limit captions that contain a noun, then it selects all of them

    # get the nouns
    nouns = question["relevantNouns"]
    # get the captions (read captions/sceneID.txt)
    sceneID = question["scene_id"]

    captionsBlip = readCaptions(f"captions/{sceneID}.txt", limit)
    captionsChat = readCaptions(f"captions/{sceneID}_chat.txt", limit)

    # interleave the two lists, accounting for the fact that one list may be longer than the other. Stick any remaining elements on the end.
    captions = []
    for i in range(max(len(captionsBlip), len(captionsChat))):
        if i < len(captionsBlip):
            captions.append(captionsBlip[i])
        if i < len(captionsChat):
            captions.append(captionsChat[i])

    # remove newlines
    captions = [caption.strip() for caption in captions]

    # split captions into `limit` sublists
    captions = [captions[i : i + limit] for i in range(0, len(captions), limit)]

    # initialize a dictionary to store the captions
    nounCaptions = {}
    for noun in nouns:
        nounCaptions[noun] = []
    # for each caption, check if it contains each noun. Perform a breadth-first search on the sublists (search the first object in each sublist, then the second object in each sublist, etc.)
    for captionList in captions:
        for i, caption in enumerate(captionList):
            for noun in nouns:
                if noun in caption:
                    nounCaptions[noun].append(caption)
                    # remove the caption from the list so that it isn't added to the dictionary again
                    captionList[i] = ""

    # remove duplicates
    for noun in nouns:
        nounCaptions[noun] = list(set(nounCaptions[noun]))

    # limit the number of captions for each noun
    for noun in nouns:
        nounCaptions[noun] = nounCaptions[noun][:limit]

    # update the question

    question["context"] = list(
        set(itertools.chain.from_iterable(nounCaptions.values()))
    )

    # open ground truth captions (GTCaptions/sceneID.txt)
    with open(f"GTCaptions/{sceneID}.txt") as f:
        captions = f.readlines()
    # remove newlines
    captions = [caption.strip() for caption in captions]

    # split captions into `limit` sublists
    captions = [captions[i : i + limit] for i in range(0, len(captions), limit)]

    # initialize a dictionary to store the captions
    nounCaptions = {}
    for noun in nouns:
        nounCaptions[noun] = []
    # for each caption, check if it contains each noun. Perform a breadth-first search on the sublists (search the first object in each sublist, then the second object in each sublist, etc.)
    for captionList in captions:
        for i, caption in enumerate(captionList):
            for noun in nouns:
                if noun in caption:
                    nounCaptions[noun].append(caption)
                    # remove the caption from the list so that it isn't added to the dictionary again
                    captionList[i] = ""

    # remove duplicates
    for noun in nouns:
        nounCaptions[noun] = list(set(nounCaptions[noun]))

    # limit the number of captions for each noun
    for noun in nouns:
        nounCaptions[noun] = nounCaptions[noun][:limit]

    question["gtCaptions"] = list(
        set(itertools.chain.from_iterable(nounCaptions.values()))
    )

    return question


def getNounsOpenAI():
    # load v1_balanced_questions_test_scannetv2.json
    with open("v1_balanced_questions_test_scannetv2.json") as f:
        questions = json.load(f)["questions"]

    retryLater = []

    q2 = []
    # for each question, prompt gpt-3.5 to return a list of important nouns
    for i, question in enumerate(tqdm.tqdm(questions)):
        # wait one second every 5 questions to avoid hitting the rate limit
        if i % 5 == 0:
            sleep(1)

        prompt = f"Return the important physical objects to scan for in this question as a comma-separated list: {question['situation']} {question['question']}"

        message = [{"role": "user", "content": prompt}]
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                max_tokens=100,
                temperature=0,
            )
        except Exception as e:
            print(e)
            retryLater.append([question, i])
            q2.append([])
            continue
        print(response["choices"][0]["message"]["content"])
        words = response["choices"][0]["message"]["content"].split(", ")
        words = [word.split(" ") for word in words]
        # flatten the list
        words = list(itertools.chain.from_iterable(words))

        # add plural and singular forms of each word
        words2 = []

        for i, word in enumerate(words):
            if inflect.singular_noun(word):
                words2.append(inflect.singular_noun(word))

            if inflect.plural_noun(word):
                words2.append(inflect.plural_noun(word))
            words2.append(word)
        words = list(set(words2))

        question["relevantNouns"] = words
        q2.append(question)

    # retry the questions that failed
    while retryLater:
        for i, (question, index) in enumerate(retryLater):
            prompt = f"Return the important objects to scan for in this question as a comma-separated list: {question['situation']} {question['question']}"

            message = [{"role": "user", "content": prompt}]
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    max_tokens=100,
                    temperature=0,
                )
            except:
                retryLater.remove([question, index])
                retryLater.append([question, index])
                q2[index] = []
                continue
            # print(response["choices"][0]["message"]["content"])
            words = response["choices"][0]["message"]["content"].split(", ")
            words = [word.split(" ") for word in words]
            # flatten the list
            words = list(itertools.chain.from_iterable(words))

            # add plural and singular forms of each word
            words2 = []

            for i, word in enumerate(words):
                if inflect.singular_noun(word):
                    words2.append(inflect.singular_noun(word))

                if inflect.plural_noun(word):
                    words2.append(inflect.plural_noun(word))
                words2.append(word)
            words = list(set(words2))

            question["relevantNouns"] = words
            q2[index] = question
            # remove the question from the list
            retryLater.remove([question, index])
    return q2


# ===================================================================================================
# LEGACY CODE
# ===================================================================================================

# load v1_balanced_questions_test_scannetv2.json
# with open("v1_balanced_questions_test_scannetv2.json") as f:
#     questions = json.load(f)["questions"]
#
# nlp = spacy.load("en_core_web_sm")
#
# updated_questions = []
#
# for question in tqdm.tqdm(questions):
#     situation = question["situation"]
#     subquestion = question["question"]
#     nounPhrases, nouns = getNouns(f"Context: {situation} Q: {subquestion}")
#     spacyNounPhrases = getNounPairs(f"Context: {situation} Q: {subquestion}", nlp)
#
#     for nounPhrase in spacyNounPhrases:
#         if nounPhrase not in nounPhrases:
#             nounPhrases.append(nounPhrase)
#
#     relevantNouns = []
#     for noun in nouns:
#         if isObject(noun):
#             relevantNouns.append(noun)
#
#     for nounPhrase in nounPhrases:
#         if not isLocationNounPhrase(nounPhrase):
#             relevantNouns.append(nounPhrase)
#
#     for nounPhrase in relevantNouns:
#         if len(nounPhrase.split()) > 1:
#             relevantNouns.append(nounPhrase.split()[-1])
#
#     for i, noun in enumerate(relevantNouns):
#         if len(noun.split()) == 1 and not isObject(noun):
#             relevantNouns.remove(noun)
#
#         if (
#             len(noun.split()) == 1
#             and inflect.singular_noun(noun)
#             and inflect.singular_noun(noun) not in relevantNouns
#         ):
#             relevantNouns.append(inflect.singular_noun(noun))
#
#     relevantNouns = list(set(relevantNouns))
#     question["relevantNouns"] = relevantNouns
#
#     updated_questions.append(question)
#
# ===================================================================================================
# END LEGACY CODE
# ===================================================================================================

updated_questions = getNounsOpenAI()

# Process each question and limit the number of captions based on nouns
updated_questions = [limit_nouns(question) for question in tqdm.tqdm(updated_questions)]

# invert question order
updated_questions = updated_questions[::-1]


with open("nouns_with_context.json", "w") as f:
    json.dump(updated_questions, f)
