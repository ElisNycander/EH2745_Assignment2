 SELECT 
    rdfid, name, time, value, sub_rdfid
 FROM
     analog_values
 INTO OUTFILE 'analog_values.csv' 
 FIELDS ENCLOSED BY '"' 
 TERMINATED BY ';' 
 ESCAPED BY '"' 
 LINES TERMINATED BY '\r\n';
 
 
  SELECT 
    rdfid, name, time, value, sub_rdfid
 FROM
     measurements
 INTO OUTFILE 'measurements.csv' 
 FIELDS ENCLOSED BY '"' 
 TERMINATED BY ';' 
 ESCAPED BY '"' 
 LINES TERMINATED BY '\r\n';

 SELECT 
    rdfid, name, region_id
 FROM
     substations
 INTO OUTFILE 'substations.csv' 
 FIELDS ENCLOSED BY '"' 
 TERMINATED BY ';' 
 ESCAPED BY '"' 
 LINES TERMINATED BY '\r\n';
