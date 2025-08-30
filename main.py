from packages import query
import argparse

def main():
    

    parser = argparse.ArgumentParser(description="Search for bugs in a repository")

    parser.add_argument('--bugs', '-b', action="store_true", help="Bug detection")
    parser.add_argument('--repo', '-r', type=str, required=True, help="Path to the repository folder")
    parser.add_argument('--query', '-q', type=str, required=True, help="Search query")


    args = parser.parse_args()

    repo = args.repo
    q = args.query
    bugs = args.bugs
    
    if bugs:
        print("With bug detection")
    query.search(repo, q, bugs=bugs)

if __name__=="__main__":
    main()
