sed -n '/```{r/,/```/p' $1 > /tmp/swap1
sed -E 's/```{.*/# %%/g' /tmp/swap1 > /tmp/swap2
sed -E 's/```//g' /tmp/swap2 > $2
cat $2
rm /tmp/swap1 /tmp/swap2
