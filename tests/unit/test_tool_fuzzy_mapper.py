from nvd_claude_proxy.translators.tool_fuzzy_mapper import FuzzyToolMapper


def test_fuzzy_tool_mapping_strict():
    valid = {"read_file", "write_file", "list_dir", "grep_search", "bash"}
    mapper = FuzzyToolMapper(valid)

    assert mapper.map_name("Read") == "read_file"
    assert mapper.map_name("Write") == "write_file"
    assert mapper.map_name("List") == "list_dir"
    assert mapper.map_name("Search") == "grep_search"
    assert mapper.map_name("Bash") == "bash"


def test_fuzzy_tool_mapping_case_insensitive():
    valid = {"readFile", "WriteFile"}
    mapper = FuzzyToolMapper(valid)

    assert mapper.map_name("readfile") == "readFile"
    assert mapper.map_name("WRITEFILE") == "WriteFile"


def test_fuzzy_tool_mapping_difflib():
    valid = {"read_file", "write_file"}
    mapper = FuzzyToolMapper(valid)

    # Close match
    assert mapper.map_name("read_fil") == "read_file"
    assert mapper.map_name("wrte_file") == "write_file"


def test_fuzzy_tool_mapping_prefix():
    valid = {"grep_search"}
    mapper = FuzzyToolMapper(valid)

    assert mapper.map_name("grep") == "grep_search"


def test_fuzzy_tool_mapping_arguments():
    valid = {"view_file", "list_dir"}
    mapper = FuzzyToolMapper(valid)

    # read_file: path -> AbsolutePath
    args = {"path": "/foo.txt", "explanation": "test"}
    mapped = mapper.map_arguments("view_file", args)
    assert mapped == {"AbsolutePath": "/foo.txt", "explanation": "test"}

    # ls: path remains path
    args = {"path": "/foo"}
    mapped = mapper.map_arguments("list_dir", args)
    assert mapped == {"DirectoryPath": "/foo"}


def test_fuzzy_tool_mapping_no_match():
    valid = {"ls", "bash"}
    mapper = FuzzyToolMapper(valid)

    assert mapper.map_name("unknown_tool") is None
