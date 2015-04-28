FILE(REMOVE_RECURSE
  "CMakeFiles/myproject.dir/myproject.cc.o"
  "myproject.pdb"
  "myproject"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang CXX)
  INCLUDE(CMakeFiles/myproject.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
