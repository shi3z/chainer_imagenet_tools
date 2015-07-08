# Usage
#
#  ls images | awk -v P=${pwd) -v src=(train.txt|test.txt) 
#
BEGIN {
  P=P"/images/"
  while(getline < src > 0){
    file[$1] = $2
  }
  close(list)
}

{
  if(file[P$1]){
    print P$1" "file[P$1]
  }
}

