# Instructions for downloading and installing INDRI

# 1. Download INDRI 5.14

https://sourceforge.net/projects/lemur/files/lemur/ 

https://sourceforge.net/projects/lemur/files/lemur/indri-5.14/

# 2. Create INDRI directory

```
mkdir indri
```

```
cd indri-5.14
```

# 3. Install compiler: gcc-5 e g++-5

```
export CC=gcc-5
```

```
export CXX=g++-5
```

```
./configure --prefix= $home/indri
```

```
make
```

```
make install
```

it creates 4 directories in $home/indri: bin, include, lib, and share 


# 4. Creating the Inverted Index

```
cp parameters_NOSW.xml indri/bin
```

```
cd indri/bin
```

```
nohup ./IndriBuildIndex par_index_NOSW.xml 2> err_index > log_index &
```

# 5. Running Queries

```
cp run_indri.sh indri/bin
```

```
cd indri/bin
```

```
nohup ./run_indri.sh $home/input_queries/topic_shift.query $home/results/topic_shift.txt &
```

