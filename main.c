#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>


const char DEFAULT_ERROR[] = "[error]";
const size_t DEFAULT_STR_SIZE = 128;

void throw_error(const char*);
void get_params(unsigned int*, unsigned int*, char* []);
unsigned int get_number(char);
char get_numerial(unsigned int);

char* convert(unsigned int, unsigned int, const char*); // P Q S
char* reverse(char*, int);


int main()
{
    unsigned int p, q;
    char *s = NULL;

    // Input
    // printf("\nEnter input:");
    get_params(&p, &q, &s);

    char *res = convert(p, q, s);
    printf("%s\n", res);

    free(s);
    free(res);
}


void throw_error(const char* msg)
{
    printf("%s", msg);
    exit(0);
}

unsigned int get_number(char c)
{
    return (c >= 'A') ? (c - 'A' + 10) : (c - '0');
}

char get_numerial(unsigned int n)
{
    // -- check???
    return (n >= 10) ? (n + 'A' - 10) : (n + '0');
}


char* reverse(char *str, int len)
{
    char *p1, *p2;

    if (! str || ! *str)
        return str;
    for (p1 = str, p2 = str + len - 1; p2 > p1; ++p1, --p2)
    {
        *p1 ^= *p2;
        *p2 ^= *p1;
        *p1 ^= *p2;
    }
    return str;
}


char* convert(unsigned int p, unsigned int q, const char* s)
{
    size_t s_len = strlen(s);

    // Convert to decimal
    unsigned long long s_10_base = 0;
    double power = pow(p, s_len);

    for (int i = 0; i < s_len ; ++i) {
        power /= p;
        s_10_base += get_number(s[i]) * power; 
    }

    // Convert to Q based
    size_t s_q_base_len = DEFAULT_STR_SIZE;
    char *s_q_base = malloc(DEFAULT_STR_SIZE);

    unsigned long long d = s_10_base,
                       r = 0;
    size_t i = 0;
    do
    {
        r = d % q;
        d = d / q;
        if(i == s_q_base_len)
        {
            s_q_base_len = i + DEFAULT_STR_SIZE;
            char* tmp_ret = realloc(s_q_base, s_q_base_len);

            if(tmp_ret == NULL)
            {
                free(s_q_base);
                throw_error(DEFAULT_ERROR);
            }

            s_q_base = tmp_ret;
        }
        s_q_base[i] = get_numerial(r);

        i++;
    } while(d > 0);

    s_q_base = reverse(s_q_base, i);
    s_q_base[i] = '\0';

    return s_q_base;
}

void get_params(unsigned int* p, unsigned int* q, char* s[])
{
    // Get P, Q
    int n;
    n = scanf("%d %d ", p, q);
    if (n != 2)
        throw_error(DEFAULT_ERROR);

    if ((*q < 2) || (*q >= *p) || (*p > 36))
        throw_error(DEFAULT_ERROR);

    // Get S
    size_t p_str_size = DEFAULT_STR_SIZE;
    char *p_str = malloc(DEFAULT_STR_SIZE);
    size_t i = 0;
    char c = EOF;

    if(p_str != NULL)
    {
        // int c = EOF; -- ???
        // read until user hits enter
        while (( c = getchar() ) != '\n' && c != EOF)
        {
            c = toupper(c);

            if ((get_number(c) > *p - 1))
            {
                free(p_str);
                throw_error(DEFAULT_ERROR);
            }

            p_str[i++] = c;

            // if reached len_max - realloc 
            if(i == p_str_size)
            {
                p_str_size = i + DEFAULT_STR_SIZE;
                char* p = realloc(p_str, p_str_size);

                // memmory request fails
                if (p == NULL)
                {
                    free(p_str);
                    throw_error(DEFAULT_ERROR);
                }

                p_str = p;
            }
        }
        p_str[i] = '\0';
    } else {
        throw_error(DEFAULT_ERROR);
    }

    *s = p_str; // library style
}
