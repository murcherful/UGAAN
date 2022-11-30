class DICT2ARGS:
    def __init__(self, _dict):
        for key in _dict.keys():
            setattr(self, key, _dict[key])

def dict2attr(x, _dict):
    for key in _dict.keys():
        setattr(x, key, _dict[key])

class ARGS_SCANNET:
    C_ARGS = {
        'chair':{
            'dis':0.03,
            'min_scale':0.5,
            'max_scale':0.8,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },
        'table':{
            'dis':0.03,
            'min_scale':0.4,
            'max_scale':0.7,
            'max_center_shift':0.1,
            'pr':False,
            'max_epoch':480,
        },
        'bookshelf':{
            'dis':0.03,
            'min_scale':0.4,
            'max_scale':0.7,
            'max_center_shift':0.1,
            'pr':False,
            'max_epoch':480,
        },
        'sofa':{
            'dis':0.03,
            'min_scale':0.4,
            'max_scale':0.7,
            'max_center_shift':0.1,
            'pr':False,
            'max_epoch':480,
        },        
    }
    def __init__(self, __c):
        dict2attr(self, self.C_ARGS[__c])
        self.lmdb_train = 'data/ws/ws_data_%s_0p5_train.lmdb' % __c 
        self.lmdb_valid = 'data/ws/ws_data_%s_0p5_test.lmdb' % __c 
        self.lmdb_test = 'data/ws/ws_data_%s_0p5_test.lmdb' % __c 
        self.lmdb_test_s3dis = 'data/s3dis_coseg'
        self.lmdb_sn = 'data/shapenet/shapenet_data_%s.lmdb' % __c
        self.log_dir = 'log/scannet/%s/' % __c
        self.res_dir = 'res_scannet'
        self.res_dir_s3dis = 'res_s3dis'
        self.batch_size = 4 
        self.input_point_number = 2048
        self.gt_point_number = 2048
        self.save_pre_epoch = 10
        self.print_pre_step = 100
        self.test_pre_step = -1


'''
min_scale and max_scale can be set to 0.7/0.9 and 0.8/1.0.
'''
class ARGS_SCANOBJ:
    C_ARGS = {
        'chair':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },
        'table':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },
        'bookshelf':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },
        'sofa':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },     
        'bed':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },     
        'pillow':{
            'dis':0.03,
            'min_scale':0.8,
            'max_scale':1.0,
            'max_center_shift':0.0,
            'pr':True,
            'max_epoch':240,
        },  
        'cabinet':{
            'dis':0.03,
            'min_scale':0.7,
            'max_scale':0.9,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },  
    }
    def __init__(self, __c):
        dict2attr(self, self.C_ARGS[__c])
        self.lmdb_train = 'data/ws/ws_data_%s_0p05_train.lmdb' % __c 
        self.lmdb_valid = 'data/ws/ws_data_%s_0p05_test.lmdb' % __c 
        self.lmdb_test = 'data/scanobjectnn/h5_files/main_split'
        self.lmdb_sn = 'data/shapenet/shapenet_data_%s.lmdb' % __c
        self.log_dir = 'log/scannet_scanobj/%s/' % __c
        self.res_dir = 'res_scanobj'
        self.batch_size = 4 
        self.input_point_number = 2048
        self.gt_point_number = 2048
        self.save_pre_epoch = 10
        self.print_pre_step = 100
        self.test_pre_step = -1



class ARGS_SCANNET_COM:
    C_ARGS = {
        'chair':{
            'dis':0.03,
            'min_scale':0.5,
            'max_scale':0.8,
            'max_center_shift':0.0,
            'pr':False,
            'max_epoch':240,
        },
        'table':{
            'dis':0.03,
            'min_scale':0.4,
            'max_scale':0.7,
            'max_center_shift':0.1,
            'pr':False,
            'max_epoch':240,
        },
        'bookshelf':{
            'dis':0.03,
            'min_scale':0.4,
            'max_scale':0.7,
            'max_center_shift':0.1,
            'pr':False,
            'max_epoch':240,
        },
        'sofa':{
            'dis':0.03,
            'min_scale':0.4,
            'max_scale':0.7,
            'max_center_shift':0.1,
            'pr':False,
            'max_epoch':120,
        },        
    }
    def __init__(self, __c):
        dict2attr(self, self.C_ARGS[__c])
        self.lmdb_test = 'data/us_gt/us_data_gt.lmdb'
        self.lmdb_sn = 'data/shapenet/shapenet_data_%s.lmdb' % __c
        self.log_dir = 'log/scannet/%s/' % __c
        self.res_dir = 'res_scannet_com'
        self.batch_size = 4 
        self.input_point_number = 2048
        self.gt_point_number = 2048
        self.save_pre_epoch = 10
        self.print_pre_step = 100
        self.test_pre_step = -1