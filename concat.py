import os
import os.path as osp

import sys
import json
import struct
import numpy as np
import matio
import argparse

import time

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--aDir', type=str, default='test/t1', help='feature dir a')
    parser.add_argument('--bDir', type=str, default='test/t2', help='feature dir b')
    parser.add_argument('--dirList', type=str, default='test/t1+test/t2+test/t3', help='dir list, split with +')
    parser.add_argument('--saveDir', type=str, default='testout', help='save dir')
    parser.add_argument('--txtlist', type=str, default='', help='txt file of paths')
    return parser.parse_args(argv)

def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature),1,4,5))
        f.write(struct.pack("%df"%len(feature), *feature))

def read_txtlist(sourceDir, path):
    pathList = []
    with open(path, 'r') as f:
        aPath = f.readline().strip()
        while aPath:
            aPath += '_feat.bin'
            pathList.append(sourceDir + aPath)
            aPath = f.readline().strip()
    return pathList


def createPath_write_bin(path, feature):
    try:
        file_dir = os.path.split(path)[0]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
    except:
        pass 
    write_bin(path, feature)
    return

def recursive_list(targetDir, outList=[]):
    for path in os.listdir(targetDir):
        if path.startswith('.'):
            continue
        fullPath = os.path.join(targetDir, path)
        if os.path.isdir(fullPath):
            recursive_list(fullPath, outList)
        else:
            outList.append(fullPath)
    return outList
        
def concat2_from_list(aDir, bDir, pathList, saveDir):
    for path in pathList:
        aPath = path
        bPath = path.replace(aDir, bDir)
        _concatedPath = path.replace(aDir, '')
        concatedPath = os.path.join(saveDir, _concatedPath)
        try:
            aFeatureVec = np.transpose(matio.load_mat(aPath))
            bFeatureVec = np.transpose(matio.load_mat(bPath))
            concatedFeatureVec = np.concatenate((aFeatureVec, bFeatureVec), axis=1)
            #print concatedPath
            createPath_write_bin(concatedPath, concatedFeatureVec)
        except:
            print 'unable to process'
    return 

def concatN_from_list(dirList, pathList, saveDir):
    sourceDir = dirList[0]
    N = len(dirList)
    errorCnt = 0
    for path in pathList:
        sourcePath = path
        tmpVec = np.transpose(matio.load_mat(sourcePath))[:, 0:-1]
        _concatedPath = path.replace(sourceDir, '')
        if _concatedPath.startswith('/'):
            _concatedPath = _concatedPath[1:]
        concatedPath = os.path.join(saveDir, _concatedPath)
        # try:
        
        for followDir in dirList[1:]:
            followPath = sourcePath.replace(sourceDir, followDir)
            untransposed = matio.load_mat(followPath)
            followVec = np.transpose(untransposed)[:, 0:-1]
            tmpVec = np.concatenate((tmpVec, followVec), axis=1)
        #pad the last label 
        tmpVec = tmpVec / np.sqrt(N)
        tmpVec = np.concatenate((tmpVec, np.expand_dims(np.transpose(untransposed)[:,-1], axis=0)), axis=1)
        
        createPath_write_bin(concatedPath, tmpVec.T)

    return 




def main(args):
    t0 = time.time()
    print('===> args:\n', args)
    
    txtlist = args.txtlist
    dirList = args.dirList.split('+')
    for i in range(len(dirList)):
        if not dirList[i].endswith('/'):
            dirList[i] += '/'

    print dirList
    saveDir = args.saveDir

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    
    if args.txtlist == '':
        featurePathList = recursive_list(dirList[0])
    else:
        featurePathList = read_txtlist(dirList[0], txtlist)
    print 'feature list done'

    #concat2_from_list(aDir, bDir, featurePathList, saveDir)
    concatN_from_list(dirList, featurePathList, saveDir)
    print 'concat done'

    t1 = time.time()
    print "time cost:", t1 - t0

    
if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
