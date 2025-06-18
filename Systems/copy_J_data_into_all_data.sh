


paths=('RUN0_1a' 'RUN1_1b' 'RUN7_2a' 'RUN2_2b' 'RUN4_3a' 'RUN2_3b' 'RUN3_4a' 'RUN4_4b' 'RUN5_5' 'RUN6_8' 'RUN13_1' 'RUN12_2');
for i in "${paths[@]}"; do

  if [ "$i" != "RUN6_8" ]; then
    #yes | cp "$i"/CS_J_NOE/*.J "$i"/all_data/;
    #yes | cp "$i"/CS_J_NOE/*.noe "$i"/all_data/;

    yes | cp "$i"/500_states/CS_J_NOE/*.J "$i"/500_states/all_data/;
    yes | cp "$i"/500_states/CS_J_NOE/*.noe "$i"/500_states/all_data/;
  fi
done;

exit 1;








