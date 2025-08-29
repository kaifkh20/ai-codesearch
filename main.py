from packages import query


def main():
    
    repo = input("Enter folder path: ").strip()
    q = input("Search query: ").strip()
    query.search(repo,q)


if __name__=="__main__":
    main()
