import time 

def test1():
    print("called first time")
    yield 

    print("called last time")


if __name__ == "__main__":
    test1()
    time.sleep(5)
    print("5 seconds elapsed...")
    test1()


