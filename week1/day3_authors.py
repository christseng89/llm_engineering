# Example books list
def example_books():
    books = [
        {"title": "Book One", "author": "Author A"},
        {"title": "Book Two", "author": "Author B"},
        {"title": "Book Three", "author": "Author C"},
        {"title": "Book Four", "author": ""},
        {"title": "Book Five"},
        {"title": "Book Six", "author": "Author A"}
    ]
    return books

# Function to get unique authors directly as a list
def get_unique_authors(books):
    seen_authors = {book.get("author") for book in books if book.get("author")}
    return list(seen_authors)

# Use the function and print authors
authors = get_unique_authors(example_books())
print('Authors:', authors)
print('Sorted Authors:', sorted(authors))  # Sorted list of unique authors