import re

exp = "v([0-9]+)([.])([0-9]+)([.])([0-9]+)(b|rc)?([0-9]+)?"
exp = re.compile(exp)

def main(args):
    print("The version is", args.version)

    with open("README.md") as f:
        s = f.read()

    s = re.sub(exp, args.version, s)

    with open("README.md", "w") as f:
        f.write(s)

    

if __name__ == '__main__':
    import argparse                                                           
    parser = argparse.ArgumentParser()                                        
    parser.add_argument("-v", "--version", type=str)                         
    args = parser.parse_args()
    main(args)