class DICT2ARGS:
    def __init__(self, _dict):
        for key in _dict.keys():
            setattr(self, key, _dict[key])

def dict2attr(x, _dict):
    for key in _dict.keys():
        setattr(x, key, _dict[key])


'''
min_scale and max_scale can be set to 0.7/0.9 and 0.9/1.0.
'''
class ARGS_SCANOBJ:
    C_ARGS = {
        'chair':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },
        'table':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },
        'bookshelf':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },
        'sofa':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },     
        'bed':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },     
        'pillow':{
            'dis':0.09,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':True,
            'max_epoch':60,
        },  
        'cabinet':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },  
        'bin':{
            'dis':0.10,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },  
        'bag':{
            'dis':0.15,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':True,
            'max_epoch':60,
        },  
        'monitor':{
            'dis':0.09,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        },  


        'box':{
            'dis':0.25,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        }, 
        'desk':{
            'dis':0.09,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        }, 
        'door':{
            'dis':0.10,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        }, 
        'sink':{
            'dis':0.10,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        }, 
        'toilet':{
            'dis':0.25,
            'min_scale':0.9,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':60,
        }, 
    }
    def __init__(self, __c):
        dict2attr(self, self.C_ARGS[__c])
        self.lmdb_train = 'data/scanobjectnn/h5_files/main_split'
        self.lmdb_valid = 'data/scanobjectnn/h5_files/main_split'
        self.lmdb_test = 'data/scanobjectnn/h5_files/main_split'
        if __c in ['table', 'chair', 'sofa', 'bookshelf', 'bed', 'monitor', 'pillow', 'cabinet', 'bin', 'bag']:
            self.lmdb_sn = 'data/shapenet/shapenet_data_%s.lmdb' % __c
        else:
            self.lmdb_sn = 'data/modelnet/modelnet_data_%s_all.lmdb' % __c
        self.log_dir = 'log/scanobj/%s/' % __c
        self.res_dir = 'res_scanobj'
        self.batch_size = 4 
        self.input_point_number = 2048
        self.gt_point_number = 2048
        self.save_pre_epoch = 10
        self.print_pre_step = 100
        self.test_pre_step = -1