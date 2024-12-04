import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import time

# Number of processes to create
import settings as s
import math

class ExperimentWorker():
    def __init__(self,experiment = None):
        self.classified_data = [] # store classified rows, which can be entered into database
        self.nbr_tries = []
        self.return_codes = []

        self.exp = experiment
        self.exp_nbr = None

    # do the actual experiment it is repeated if an error occurs during execution
    def do_experiment(self,data):
        try:
            x = self.exp(data)
            if x is None:
                return 1
            self.classified_data.append(x)
            return 0
        except Exception as e:
            print(e)
            return 1

    # do single experiment
    def run_single_experiment(self, data_list):
        nbr_try = 1
        while nbr_try <= 3:
            ret = self.do_experiment(data_list)
            if ret == 0:
                self.nbr_tries.append(nbr_try)
                return 0
            nbr_try += 1
        self.nbr_tries.append(nbr_try)
        return 1

    # main methods starts calculation of single instances
    def run(self, data_pair):
        dat_nbr = data_pair[1]
        assert isinstance(dat_nbr, int)
        self.exp_nbr = dat_nbr
        data_to_calculate = data_pair[0]
        if len(data_to_calculate) == 0:
            return self
        for data in data_to_calculate:
            ret = self.run_single_experiment(data)
            self.return_codes.append(ret)
        return self

class MCCalc():

    def __init__(self):
        self.final_calculated_data = None

    def split_text(self, argument_list, splitter = None, enumerate = True): # used or splitting the required list
        if splitter is None:
            splitter = s.NBR_PROCESSES
        split_size = len(argument_list) / splitter
        split_size = math.ceil(split_size)
        if enumerate:
            data_to_return_list = [(argument_list[i * split_size: (i + 1) * split_size], i) for i in range(splitter)]
            test_enum = [x[0] for x in data_to_return_list]
            test_enum_tmp = []
            for x in test_enum :
                test_enum_tmp.extend(x)
            assert len(test_enum_tmp) == len(argument_list)
            return data_to_return_list

        data_to_return_list = [argument_list[i * split_size: (i + 1) * split_size] for i in range(splitter)]
        test_enum_tmp = []
        for x in data_to_return_list :
            test_enum_tmp.extend(x)
        assert len(test_enum_tmp) == len(argument_list)
        return data_to_return_list

    def split_text_targer(self, argument_list, splitter = None, enumerate = True): # used or splitting the required list
        if splitter is None:
            splitter = s.NBR_PROCESSES
        split_size = len(argument_list) / splitter
        split_size = math.ceil(split_size)
        if enumerate:
            data_to_return_list =  [([argument_list[i * split_size: (i + 1) * split_size]],i) for i in range(splitter)]
            test_enum = [x[0][0] for x in data_to_return_list]
            test_enum_tmp = []
            for x in test_enum:
                test_enum_tmp.extend(x)
            assert len(test_enum_tmp) == len(argument_list)
            return data_to_return_list

        data_to_return_list = [[argument_list[i * split_size: (i + 1) * split_size]] for i in range(splitter)]
        test_enum_tmp = []
        for x in data_to_return_list :
            test_enum_tmp.extend(x)
        assert len(test_enum_tmp) == len(argument_list)
        return data_to_return_list


    def run_mult(self, exp, data_split):
        start_time = time.time()

        # Create a pool of processes
        pool = multiprocessing.Pool(processes=s.NBR_PROCESSES)

        # Process each chunk of text in parallel
        processed_chunks = pool.map(exp, data_split)

        # Wait for all processes to complete
        pool.close()
        pool.join()

        # Combine processed chunks
        self.processed_argus = list(sorted(processed_chunks, key = lambda x: x.exp_nbr))

        info_mp_process = [x.nbr_tries for x in self.processed_argus]
        return_codes_tmp = [x.return_codes for x in self.processed_argus]
        return_codes = []
        for x in return_codes_tmp:
            return_codes.extend(x)

        print('Nbr of tries per process',info_mp_process)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")

        calculated_data = [x.classified_data for x in self.processed_argus]

        return calculated_data, return_codes

    def get_data(self):
        list_flattended = []
        for x in self.processed_argus:
            list_flattended.extend(x.classified_data)
        return list_flattended



if __name__ == '__main__':
    data_test = [x for x in range(3)]
    data_test_result = MCCalc().split_text_targer(data_test)
    mewo = 1