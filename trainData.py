from __future__ import division


class Dataset(object):
    def __init__(self, opt, dataset):
        self.data_set = dataset
        self.nums = opt.validation

    def __getitem__(self, index):
        return (self.data_set['ID'], self.data_set['IM'],
                self.data_set['md'][index]['train'], self.data_set['md'][index]['test'],
                self.data_set['md_p'], self.data_set['md_true'],
                self.data_set['independent'][0]['train'],self.data_set['independent'][0]['test'])

    def __len__(self):
        return self.nums



