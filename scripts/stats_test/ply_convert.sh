#!/bin/bash
outfile='limit_vertices.ply'
echo ply > $outfile
echo 'format ascii 1.0' >> $outfile
numverts=$(wc -l mesh_dump.dat | cut -d ' ' -f 1)
echo "element vertex ${numverts}" >> $outfile
echo "property float x" >> $outfile
echo "property float y" >> $outfile
echo "property float z" >> $outfile
echo "end_header" >> $outfile
cat mesh_dump.dat >> $outfile
