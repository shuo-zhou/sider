#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:22:15 2018

@author: Shuo Zhou, the University of sheffield

USAGE: python <PROGRAM> options
ACTION: read options from cmd input
OPTIONS:
    -h: print help
    -a: specify the atlas of ABIDE time courses
    -d: specify the data for experiemnt
    -m: specify the measure kind for brain connectome
    -p: specify the transfer learning problem
    -v: load vectorized data
"""

import sys, os
import getopt


class commandline:
    def __init__(self):
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:k:a:c:m:l:svh')
        opts = dict(opts)        
        # default do not submit jod to HPC
        
        self.data = 'openfmri'
        self.atlas = 'cc200'
        self.kind = 'tangent'
        self.prob = 1
        self.vectorize = False
        self.swap = False
        self.kfold = 5
        self.kernel = 'linear'
        self.loss = 'hinge'
        
        if '-h' in opts:
            self.printHelp()
            
        if '-a' in opts:
            self.atlas = opts['-a']
        
        if '-d' in opts:
            self.data = opts['-d']
        
        if '-m' in opts:
            self.kind = opts['-m']
        
        if '-p' in opts:
            self.prob = int(opts['-p'])

        if '-c' in opts:
            self.kfold = int(opts['-c'])

        if '-k' in opts:
            if opts['-k'] in ['linear', 'rbf', 'poly']:
                self.kernel = opts['-k']
            else:
                print('Invalid kernel')
                sys.exit()

        if '-l' in opts:
            if opts['-l'] in ['hinge', 'ls']:
                self.loss = opts['-l']
            else:
                print('Invalid loss type')
                sys.exit()
            
        if '-s' in opts:
            self.swap = True
            
        if '-v' in opts:
            self.vectorize = True

    def printHelp(self):
        helpinfo = __doc__.replace('<PROGRAM>', sys.argv[0], 1)
        print(helpinfo)
        print(sys.stderr)
        sys.exit()
