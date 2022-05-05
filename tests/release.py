def main(args):
    print("The version is", args.version)

if __name__ == '__main__':
    import argparse                                                           
    parser = argparse.ArgumentParser()                                        
    parser.add_argument("-v", "--version", type=str)                         
    args = parser.parse_args()
    main(args)