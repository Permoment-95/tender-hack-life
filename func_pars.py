from nltk.tokenize import word_tokenize
from pyaspeller import YandexSpeller
from transliterate import translit
import regex as re
import pandas as pd
import numpy as np
from pymystem3 import Mystem
import spacy
from spacy.lang.ru.examples import sentences
import json

nlp = spacy.load("ru_core_news_md")

# data = pd.read_csv('table_data.csv', sep=',')
#
# with open('obscene_corpus.txt', 'r') as f:
#     file = f.readlines()
#
# with open('forbidden_list.txt', 'r') as f:
#     fb_file = f.readlines()

# vocab_obscene = vocab_lower(file)
# vocab_forbidden = vocab_lower(fb_file)

#preprocessing of vocabulary of obscene words
def vocab_lower(vocabulary: list):
    list_of_lower_words = []
    for word in vocabulary:
        list_of_lower_words.append(word.lower().replace('\n',''))
    return list_of_lower_words


# Токенизация текста
def tokenize_text(text: str):
    tokens_list = word_tokenize(text)
    return tokens_list


# Приведение текста к нижнему регистру
def lower_text(text: str):
    return (text.lower())


# Транслитерация текста (латиница -> кириллица)
def translit_chars(text: str):
    return (translit(text, 'ru', reversed=False))


# Удаление знаков препинания
def re_gex(text: str):
    nabor = re.compile(r'[.+-,!@"*#$%^&)(|\/?=_:;]')
    text_clean = nabor.sub(r' ', text)
    return text_clean


# Функция препроцессинга текста
def text_preprocess(text: str):
    '''Функция препроцессинга текста: токенизация,
        приведение к нижнему регистру,
        транслитерация и удаление знаков'''

    tokens_list = tokenize_text(lower_text(translit_chars(re_gex(text))))
    return tokens_list


# Исправление опечаток
def correct_typos(tokens_list: list):
    '''Функция принимает на вход список токенов и заменяет грамматически неверные слова на верные.
    На выходе получается список той же длины но со словами без опечаток'''

    pel = YandexSpeller()

    correct_tokens_list = []
    for word in tokens_list:
        correct_tokens_list.append(pel.spelled(word))

    assert len(tokens_list) == len(correct_tokens_list)

    return correct_tokens_list


# проверка на наличие опечаток в тексте
def check_typos(text: str):
    '''В данной функции сравнивается список токенов исходного текста
    со списком токенов прогнанных через функция проверки опечаток:
    в случае совпадения (успеха) возвращается 0, в случае несоответствия возвращается 1,
    что означает, что в тексте присутствуют опечатки'''

    tokens_list = text_preprocess(text)
    correct_list = correct_typos(tokens_list)
    for i, j in zip(tokens_list, correct_list):
        if j != i:
            return True
        else:
            return False


# проверка на наличие обсценной лексики
def check_obscene(text: str, vocab_of_obscene: list):
    '''Функция ищет соответствия токенов из текста словам из словаря обсценной лексики.'''

    tokens_list = text_preprocess(text)
    for token in tokens_list:
        if token in vocab_of_obscene:
            return True
    return False


# проверка на наличие слов, обозначающих запрещенный товар к продаже
def check_forbidden_goods(text: str, vocab_of_forbidden_goods: list):
    '''Функция ищет соответствия токенов текста словам из списка наименований запрещенной продукции'''

    tokens_list = text_preprocess(text)
    model = Mystem()
    lemmas = []
    for word in tokens_list:
        a = model.lemmatize(word)
        if a[0] in vocab_of_forbidden_goods:
            return True
    return False


# Проверка пропусков в текстовых полях для полной формы
def check_gaps(names_values_list: list):
    lst = [None, '', ' ', '\n', '\t']
    for text_name in names_values_list:
        if text_name == (i for i in lst):
            return True
        else:
            return False


# Проверка пропусков в текстовых полях для упрощенной формы
def checks_gaps_simple(text: str):
    lst = [None, '', ' ', '\n', '\t']
    if text_name == (i for i in lst):
        return True
    else:
        return False


# Проверка на наличие в форме не менее 4-х характеристик
def check_lenght(lenght_chars: int):
    '''Обязательных полей на данный момент 3,
        в заявке необходимо указать дополнительно не менее 4 характеристик товара.
        При изменении значений параметров необходимо изменить в функции их сумму.'''

    if lenght_chars >= 7:
        return True
    else:
        return False


# Проверка на наличие в поле "Описание" количества символов не менее 50
def check_lenght_simple(text: str):
    if len(text) <= 50:
        return True
    else:
        return False

#Функция чтения из датафрема поля с характеристиками товара
def data_extract(data):
    char_dict = {}
    list_of_chars = json.loads(data['Исходные характеристики'].values[0])
    for char in range(len(list_of_chars)):
        char_dict[char] = list_of_chars[char]
    return char_dict


# Функция-агрегатор тестов
def checks(text: str, form_type: str,
           vocab_of_obscene: list,
           vocab_of_forbidden_goods: list,
           lenght_chars: int = None,
           names_values_list: list = None):
    '''В данной функции проводятся тесты для двух видов заявок: Полная и Упрощення.
        Разница в обработке входных данных.
        Проводимые тесты:
                - на наличие опечаток;
                - на наличие обсценной лексики;
                - на наличие наименований запрещенных товаров;
                - на пропуски в текстовых полях;
                - на количество указанных характеристик товара'''

    test_typos = check_typos(text)
    test_obscene = check_obscene(text, vocab_of_obscene)
    test_forbidden = check_forbidden_goods(text, vocab_of_forbidden_goods)

    if form_type == ['Полная']:
        test_ln = check_lenght(lenght_chars)
        test_gaps = check_gaps(names_values_list)

    else:
        test_ln = check_lenght_simple(text)
        test_gaps = check_gaps(text)

    sum_list = {'number_of_chars': test_ln,
                'gaps_in_fields': test_gaps,
                'typos_in_text_fields': test_typos,
                'obscene_in_text_fields': test_obscene,
                'forbidden_goods': test_forbidden}

    return sum_list


# Аггрегирующая функция для всех тестов
def form_checking_tests(data, vocab_of_obscene: list, vocab_of_forbidden_goods: list):
    form_type = list(data['Форма заявки'])

    if form_type == ['Полная']:
        chars = data_extract(data)
        text_all_chars = []
        names_values_list = []
        for v in chars.values():
            text = str(v['name']) + ' ' + str(v['value'])
            names_values_list.append(v['name'])
            names_values_list.append(v['value'])
            text_all_chars.append(text)

        text_all_chars_txt = ','.join(text_all_chars)
        lenght_chars = len(chars)
        results = checks(text_all_chars_txt, form_type, vocab_obscene, vocab_forbidden, lenght_chars, names_values_list)

    elif form_type == ['Упрощенная']:
        text = str(data['Описание'])
        results = checks(text, form_type, vocab_obscene, vocab_forbidden)

    else:
        raise KeyError

    # проводим тесты для наименования товара
    good_name = str(data['Наименование'])
    test_name_plur = check_plural(good_name)
    test_text_isenglish = isEnglish(good_name)
    test_name_typos = check_typos(good_name)
    test_name_obscene = check_obscene(good_name, vocab_of_obscene)
    test_name_forbidden = check_forbidden_goods(good_name, vocab_of_forbidden_goods)

    tests_list = [test_name_plur, test_text_isenglish, test_name_typos, test_name_obscene, test_name_forbidden]
    tests_names = ['plural_good_name', 'english_good_name', 'typos_in_good_name',
                   'obscene_in_good_name', 'forbidden_good']

    for name, res in zip(tests_names, tests_list):
        results[name] = res

    return results


def text_results(results: dict, provider_comments: dict, moderator_comments: dict):
    '''Функция ищет соответствия итогов результатов между словарями
    для конечного вывода для поставщика и модератора'''

    result_comment_for_provider = []
    result_comment_for_moderator = []

    for k, v in results.items():
        if v == 1:
            result_comment_for_provider.append(provider_comments[k])
            result_comment_for_moderator.append(moderator_comments[k])

    if len(result_comment_for_provider) != 0:
        result_comment_for_provider.append(provider_comments['comment_neg'])
    else:
        result_comment_for_provider.append(provider_comments['comment_pos'])
    return (
        f'Результат проверки заявки для поставщика: {result_comment_for_provider}. Результат проверки заявки для модератора: {result_comment_for_moderator}')

