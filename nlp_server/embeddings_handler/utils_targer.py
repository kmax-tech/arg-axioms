from targer_api import (
    ArgumentSentences, ArgumentLabel, ArgumentTag, analyze_text
)
from targer_api.constants import DEFAULT_TARGER_API_URL
from typing import FrozenSet
import settings as s
import multi_experiments.multiproc as mc
import time
import sys
TARGER_MODEL: str = "tag-webd-fasttext"
TARGER_MODELS: FrozenSet[str] = frozenset({TARGER_MODEL})
from loguru import logger
def build_argument(tagged_parts: ArgumentSentences):
    argu_sentences = [tag.token for tag in tagged_parts if (_is_claim(tag) or _is_premise(tag))  and tag.probability > 0.5]

    return ' '.join(argu_sentences)

def _is_claim(tag: ArgumentTag) -> bool:
    return (
            tag.label == ArgumentLabel.C_B or
            tag.label == ArgumentLabel.C_I or
            tag.label == ArgumentLabel.MC_B or
            tag.label == ArgumentLabel.MC_I
    )

def _is_premise(tag: ArgumentTag) -> bool:
    return (
            tag.label == ArgumentLabel.P_B or
            tag.label == ArgumentLabel.P_I or
            tag.label == ArgumentLabel.MP_B or
            tag.label == ArgumentLabel.MP_I
    )

def get_targer_annotation(document,task_info=None):
    targer_own_sentencizer = False
    if task_info is not None:
        targer_own_sentencizer = task_info.get(s.TASK_INFO_TARGER_OWN_SENTENCIZER,False)

    if targer_own_sentencizer:
        data = get_targer_annotation_full_document(document)

    else:
        data = get_targer_annotation_single_sentence_control(document)

    assert isinstance(data,list)

    data = [item for item in data if filter_targer(item)]
    if len(data) == 0:
        data = [" "]
    return data

def filter_targer(text):
    if len(text) <= 5:
        return False
    return True

def get_targer_annotation_full_document(document) :
    final_annotations = []
    if isinstance(document,list):
        document = ' '.join(document)

    analyzed_doc = call_targer_api(document)
    if analyzed_doc is None :
        return final_annotations

    argument_sentences = [build_argument(tagged_parts) for tagged_parts in analyzed_doc[TARGER_MODEL]]
    final_annotations.extend(argument_sentences)
    return final_annotations

def get_targer_annotation_single_sentence_control(document):
    assert isinstance(document, list)

    if s.NBR_PROCESSES: # control multiprocessing
        final_annotations = []
        mccalc = mc.MCCalc()
        mp_calc_method = mc.ExperimentWorker(get_targer_annotation_single_document_sentences)
        data_targer = mccalc.split_text_targer(document)
        all_data_targer, error = mccalc.run_mult(mp_calc_method.run, data_targer)
        if 1 in error:
            logger.error(f"Error in Targer API: {error}")
            sys.exit(1)
        for x in all_data_targer:
            assert len(x) == 1
            final_annotations.extend(x[0])

    else:
        final_annotations = get_targer_annotation_single_document_sentences(document)

    return final_annotations


def call_targer_api(document) :
    current_try = 0
    analyzed_doc = None

    while current_try < s.TARGER_TOTAL_TRIES :
        try :
            analyzed_doc = analyze_text(document, model_or_models=TARGER_MODELS, api_url=DEFAULT_TARGER_API_URL)
            break
        except ValueError as e :
            logger.error(f"Error in Targer API: {e}, Sentence: {document}")
            current_try += 1
    return analyzed_doc


def get_targer_annotation_single_document_sentences(document):
    final_annotations = []

   # raise Exception("This function is not implemented")
    for sentence in document:

        analyzed_doc = call_targer_api(sentence)
        if analyzed_doc is None:
            continue

        argument_sentences = [build_argument(tagged_parts) for tagged_parts in analyzed_doc[TARGER_MODEL]]
        if len(argument_sentences) > 1:
            logger.info(f"Multiple arguments detected in sentence: {sentence}")
        argument = ' '.join(argument_sentences)
        if argument.isspace():
            argument = ""
        final_annotations.append(argument)
    return final_annotations

if __name__ == '__main__':

    a = ['to begin i would  f3 3like to review a few points from the last few posts that have been made by both massrebut and i. after that, then i will end with my conclusion.', 'my argument will reiterate and add new and different points to bring to focus when looking at single-sex schools versus co-educational schools.', 'if massrebut was not implying or assuming that "single-sex schools are better for education," in the first statement of his previous post, i ask myself why he remains in the debate after negating his own line of logic.', 'looking at massrebut"s cited debate point (along with the title of our entire debate)', 'massrebut can realize the forfeit of his argument with the statement of "no where have i implied or assumed or even suggested that single sex schools are better for education".', 'when i stated, "there are no jobs that only hire one sex," this was an example to show that single-sex education is not the best for students because it does not prepare students for the co-ed work world.', 'to reiterate: there is not "one job" that accepts only males or only females.', 'two main purposes of us education are learning basic knowledge and socialization skills.', 'the working world includes interaction with both sexes.', 'in single-sex schools, the individual is only exposed to and socialized with one group.', 'students in single-sex schools are therefore prepared to work and cooperate with only one gender.', 'therefore, single-sex schools do not prepare students for the world of employment.', 'co-educational schools have the two purposes fulfilled (learning a basis of knowledge and socialization), whereas single-sex schools only include the knowledge base.', '"boys and girls are interested in each other and define themselves by it as stated by it as stated by dragonrule029', 'but they could try for the football team or for the musical for a million other reasons not just because they are interested.', 'they could also be doing it to impress someone or get the attention or to even become friends with someone, which proves my point correct which stated that students have different driving factors in co-ed schools and they deviate from their true paths just because the other sex is present.', 'even i agree that "not everything that is done in schools is meant to "show-off"" but is certainly influenced by the other sex and even that depends on how much interested they are or how much they like and want to impress each other which is not a good learning environment for them."', 'we have agreed that not all actions are done to show off, according to the quoted statement above.', 'although, i would like to point out that in massrebut"s first argument he said "every action of the student is because he wants to impress or show off to someone he likes in the school" and in the quotations above massrebut says: "even i agree that "not everything that is done in schools is meant to "show-off""" which is therefore contradicting and negates that part of the argument.', 'regarding influencing the opposite sex, since i have already negated where that statement began, massrebut can decide if he would like to pick up that piece of the argument again or not.', '----- "when dragonrule029 talks about homosexuality and how it will cause the same level of distraction as it will in co - education schools and students will be distracted no matter where they are, well this view is completely wrong.', 'well, because not all students are homosexual and there is a 1% chance that one or two students will be and', 'so how will this make other students distracted, the ones who are not involved and the ones who do not care?', 'in co-ed schools, everyone would be looking at each other and the level of distraction in these schools is incomparable to single sex schools and to homosexual students.', 'and what are the chances that there are not homosexuals in co-ed schools?"', 'this is yet another interpretation that was taken incorrectly, not to mention the fact that more than 1% of the student body could be homosexual or another type of sexual attraction.', 'studies have shown that 12% of students identify as homo']
    #s.NBR_PROCESSES = 4
    #start_time = time.time()
    #x = get_targer_annotation(a)
    #end_time = time.time()
    #execution_time = end_time - start_time
    #print(f"Execution time Mult: {execution_time} seconds")

    s.NBR_PROCESSES = 0
    start_time = time.time()
    y = get_targer_annotation(a)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time Single: {execution_time} seconds")

    #assert len(x) == len(y)
    #assert x == y
    task = {s.TASK_INFO_TARGER_OWN_SENTENCIZER : True}
    start_time = time.time()
    y = get_targer_annotation(a,task_info=task)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time Full Document: {execution_time} seconds")

    b ="hello, this is my first debate. i wish good luck to my opponents.commenters please advice me in the comments section and thank you!so let's get started.point 1:school uniform can be a indication of the student's pride and loyalty of the school. by wearing school uniform, students will feel proud of their school. however, this only applies if the school is reputable as if the school is not reputable, the school uniform would indicate that the student is not a good student due to his school.point 2: school uniforms help teachers and staff to identify students. this is especially useful during school trips. imagine what would happen if school uniforms were banned, how would the teachers identify their students during mishaps and accidents? besides, it is also easier for a teacher to do a head count of the students as most of the students would be wearing the same type of clothes. compared to students wearing different colored clothes with different designs , students wearing the school uniform would be certainly more easily recognized.point 3: wearing a school uniform will change a student's mindset and attitude towards their study. psychologist have proven that wearing the school uniform will cause a student to be more serious compared to wearing home clothes. thus, wearing school uniform would enable a student to focus more easily during lesson.point 4: school uniform can improve the image of a school. as the saying goes, ' the clothes makes the man' by wearing a smart standardized uniform, a student will look like a studious scholar, by wearing a shirt with a punk-like design, the student will look like a hooligan, while wearing a shirt with a cartoon design would make the student look childish and immature. the clothes of what we wear reflects who we are,even if some do not notice. for example, if we wear a tank top and yellow pants with pink polka dots to a job interview, you would be perceived as a foolish person with a can't be bothered attitude. thus, if a group of students wears a class uniform, the school would be seen as organized and would of course make a better impression than a school with students wearing clothes of different designs.point 5: school uniform can reduce the bullying of many students. bullying has been a common issue in schools due to many factors. one of the factor is the clothing. wearing inappropriate or abusive clothes can lead to a disruption in class, and also, in more serious cases, a fight. for example, if a student wore a shirt to school with the words 'chinese suck', the chinese of the school would feel offended and will most probably find a way to have revenge. some of them might even perhaps commit self harm or try to engage in a fight with the student wearing the repulsive shirt. likewise, if outsiders see a student wearing an abusive shirt, and identify the student, it would bring a bad image to the school. therefore, cases of bullying can be reduced if students wear the same uniform.point 6: due to poverty, the clothes of a poor pupil might be torn, worn or dirty. this may lead to the discrimination of other students. if all students wore the same uniform, less people would discriminate students due to their clothes as their first impression on others would be based more on the mindset , personality and attitude of the person rather than his clothes.in conclusion, the uniform may have disadvantages, however, the benefits of a uniform outweighs the negative effects of a uniform, therefore, the uniform should not be banned."
    y = get_targer_annotation(a,task_info=task)
