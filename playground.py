from pprint import pprint


def formatted_scored_sim_search_by_cos(index: str, query: str, **search_args) -> str:
    k = search_args.pop("k", 4)
    print(f"Индекс: {index}, Запрос: {query}")
    print("k =", k)
    pprint(search_args, sort_dicts=False)
    print("====================")

    return f"Индекс: {index}, Запрос: {query}"


def _multi_async_search(query: str, indexes: list, search_function, **search_args) -> list:
    return [search_function(index, query, **search_args) for index in indexes]


def process_query(query: str, indexes: list, search_function, **search_args):
    return _multi_async_search(query, indexes, search_function, **search_args)


if __name__ == "__main__":
    indexes = ["lalalalala-1", "lalalalala-2", "lalalalala-3"]
    query = "lolololo?"
    results = process_query(query, indexes, formatted_scored_sim_search_by_cos)
    for result in results: print(result)
