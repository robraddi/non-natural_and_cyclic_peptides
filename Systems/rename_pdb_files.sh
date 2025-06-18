

#path='RUN0_1a'
#path='RUN12_2'
#path='RUN13_1'
path='RUN1_1b'
path='RUN2_2b'
path='RUN2_3b'
#path='RUN3_4a'
path='RUN4_3a'
#path='RUN4_4b'
#path='RUN5_5'
path='RUN6_8'
path='RUN7_2a'



#find $path -type f -name '*.pdb' | while read file; do
#
#    # Extracting the cluster and frame numbers
#    cluster=$(echo "$file" | sed -n 's/.*_cluster\([0-9]*\).*/\1/p')
#    frame=$(echo "$file" | sed -n 's/.*_frame\([0-9]*\).*/\1/p')
#
#    # Constructing the new filename
#    dir=$(dirname "$file")
#    newname="${dir}/cluster${cluster}_frame${frame}.pdb"
#
#    # Renaming the file
#    #echo "$file" "$newname"
#    mv "$file" "$newname"
#done


find "$path" -type f -name '*.pdb' | while read file; do

    # Extracting the cluster and frame numbers
    cluster=$(echo "$file" | sed -n 's/.*_cluster\([0-9]*\).*/\1/p')
    frame=$(echo "$file" | sed -n 's/.*_frame\([0-9]*\).*/\1/p')

    # Constructing the new filename
    dir=$(dirname "$file")
    base_newname="cluster${cluster}_frame${frame}.pdb"
    newname="${dir}/${base_newname}"

    # Check if the new filename already exists
    counter=1
    while [ -e "$newname" ]; do
        newname="${dir}/${base_newname%.pdb}_$counter.pdb"
        counter=$((counter + 1))
    done

    # Renaming the file
    mv "$file" "$newname"
done





