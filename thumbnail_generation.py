# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:52:07 2017

@author: Dustin
"""

import logging, json
from queue import Queue
from threading import Thread
from time import time

import generation_functions as gf
from PIL import Image
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class ImageWorker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            card_number = self.queue.get()
            try:
                t0 = time()
                current_card = gf.CallImage(set_name, card_number, lut)
                current_card_directory = gf.GetCurrentCardDirectory(current_card)
                Path(current_card_directory).mkdir(parents=True, exist_ok=True)

                im = gf.PullImage(current_card)
                # numFilesPreviouslyDone = gf.GetNumFilesInDir(gf.GetCurrentCardDirectory(current_card))

                for i in range( row_per_card):
                    im1 = gf.DirtyImage(im)
                    img = Image.fromarray(im1, 'RGB')
                    img.save(gf.DirtyCardFileName(current_card) + "_" + str(i) + '.png')

                print( gf.DirtyCardFileName(current_card) + ' is complete! ' +"(" + str(card_number) + "/" + str(cards_in_set) + ") " + str(time() - t0) + ' seconds')
            finally:
                self.queue.task_done()

if __name__ == '__main__':
    ts = time()
    
    row_per_card = 60
    set_name = "2xm"
    queue = Queue()

    with open('lut.txt') as f:
        lut = f.read().splitlines()

        cards_in_set = len(lut)

        for x in range(24):
            worker = ImageWorker(queue)
            worker.daemon = True
            worker.start()
        
        for card_number in (number for number in range(cards_in_set)):
            logger.info('Queueing {}'.format(card_number))
            queue.put(card_number)

        queue.join()
        logging.info('Took %s seconds', time() - ts)
        logging.info('Took %s minutes', (time() - ts) /60)