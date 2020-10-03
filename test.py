'''
HW4 is to be written in a file called classify.py with the following interface:

create_vocabulary(training_directory: str, cutoff: int)
create_bow(vocab: dict, filepath: str)
load_training_data(vocab: list, directory: str)
prior(training_data: list, label_list: list)
p_word_given_label(vocab: list, training_data: list, label: str)
train(training_directory: str, cutoff: int)
classify(model: dict, filepath: str)

'''
__author__ = 'cs540-testers'
__credits__ = ['Saurabh Kulkarni', 'Alex Moon', 'Stephen Jasina',
               'Harrison Clark']
version = 'V0.0'

from classify import train, create_bow, load_training_data, prior, \
    p_word_given_label, classify, create_vocabulary
import unittest


class TestClassify(unittest.TestCase):

    # create_vocabulary(training_directory: str, cutoff: int)
    # returns a list
    def test_create_vocabulary(self):
        vocab = create_vocabulary('./EasyFiles/', 1)
        expected_vocab = [',', '.', '19', '2020', 'a', 'cat', 'chases', 'dog',
                'february', 'hello', 'is', 'it', 'world']
        self.assertEqual(vocab, expected_vocab)

        vocab = create_vocabulary('./EasyFiles/', 2)
        expected_vocab = ['.', 'a']
        self.assertEqual(vocab, expected_vocab)

    # create_bow(vocab: dict, filepath: str)
    # returns a dict
    def test_create_bow(self):
        vocab = create_vocabulary('./EasyFiles/', 1)
        check1 = create_bow(vocab, './EasyFiles/2016/1.txt')

    # load_training_data(vocab: list, directory: str)
    # returns a list of dicts
    def test_load_training_data(self):
        vocab = create_vocabulary('./EasyFiles/', 1)
        check1 = load_training_data(vocab, './EasyFiles/')

    # prior(training_data: list, label_list: list)
    # returns a dict mapping labels to floats
    # assertAlmostEqual(a, b) can be handy here
    def test_prior(self):
        vocab = create_vocabulary('./corpus/training/', 2)
        training_data = load_training_data(vocab, './corpus/training/')
        check1 = prior(training_data, ['2020', '2016'])

    # p_word_given_label(vocab: list, training_data: list, label: str)
    # returns a dict mapping words to floats
    # assertAlmostEqual(a, b) can be handy here
    def test_p_word_give_label(self):
        vocab = create_vocabulary('./EasyFiles/', 1)
        training_data = load_training_data(vocab, './EasyFiles/')
        check1 = p_word_given_label(vocab, training_data, '2020')
        check2 = p_word_given_label(vocab, training_data, '2016')

    # train(training_directory: str, cutoff: int)
    # returns a dict
    def test_train(self):
        check1 = train('./EasyFiles/', 2)

    # classify(model: dict, filepath: str)
    # returns a dict
    def test_classify(self):
        model = train('./corpus/training/', 2)
        check1 = classify(model, './corpus/test/2016/0.txt')


if __name__ == '__main__':
    print('Tester %s' % version)
    print('Reference runtime: 0.383s')
    unittest.main()
