from time import sleep
import sys
import threading

class Frequent_flush(threading.Thread):
    def __init__(self, delay):
        threading.Thread.__init__(self)
        self.delay = delay
        self.daemon = True

    def run(self):
        while True:
            sys.stdout.flush()
            sleep(self.delay)


def main():
    delay = 1
    n = 15

    flushThread = Frequent_flush(delay)
    flushThread.start()

    for i in range(n):
        print("Second ", i)
        sleep(delay)

if __name__ == '__main__':
    main()
