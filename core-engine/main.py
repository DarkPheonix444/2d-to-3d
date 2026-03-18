from controller import CoreEngine

control=CoreEngine()

def main():
    path="core-engine/floor.jpg"
    result=control.process(path)
    print(result)

main()
